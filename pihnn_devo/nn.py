import torch
import math


class enriched_PIHNN_devo(DD_PIHNN):
    """
    This PIHNN class builds on the enriched_PIHNN class from Matteo Calafà's pihnn package
    """

    def __init__(
        self,
        PDE,
        units,
        boundary,
        material={"lambda": 1, "mu": 1},
        activation=torch.exp,
        has_bias=True,
    ):

        super(enriched_PIHNN_devo, self).__init__(
            PDE, units, boundary, material, activation, has_bias
        )
        if PDE not in ["km", "km-so"]:
            raise ValueError(
                "Enriched PIHNNs such as this can be used only for linear elasticity problems."
            )
        if boundary.dd_partition is None:
            raise ValueError(
                "Enriched PIHNNs such as this can be used only with DD partitioning."
            )
        self.enrichment = boundary.enrichment
        self.cracks = boundary.cracks
        self.boundary = boundary

        self.z1 = torch.nn.Parameter(
            torch.tensor(self.cracks[0].P1, dtype=torch.cdouble, device=device)
        )
        self.z2 = torch.tensor(self.cracks[0].P2, dtype=torch.cdouble, device=device)

        sif = []
        for crack in self.cracks:
            for tip in crack.tips:
                sif.append(tip.initial_sif)
        if self.enrichment == "williams":
            self.sif = torch.nn.Parameter(torch.tensor(sif, device=device))

        elif self.enrichment == "rice":
            self.sif = 0 * torch.tensor(sif, device=device)
            self.has_crack = torch.zeros(
                [self.n_domains], dtype=torch.bool, device=device
            )
            self.crack_is_internal = torch.zeros_like(self.has_crack, device=device)
            self.crack_coords = torch.zeros(
                [self.n_domains, 1], dtype=torch.complex128, device=device
            )
            self.crack_angle = torch.zeros(
                [self.n_domains, 1], dtype=torch.double, device=device
            )
            self.crack_a = torch.zeros_like(self.crack_angle, device=device)
            for crack in self.cracks:
                d = crack.rice["domain"]
                self.has_crack[d] = 1
                self.crack_is_internal[d] = crack.rice["is_internal"]
                self.crack_coords[d] = crack.rice["coords"]
                self.crack_angle[d] = crack.rice["angle"]
                self.crack_a[d] = crack.length / 2

    def update_boundary(self):

        crack = self.boundary.cracks[0]

        z1 = self.z1
        z2 = self.z2

        crack.P1 = z1
        crack.P2 = z2

        crack.tips = []
        crack.add_crack_tip(0)
        crack.add_crack_tip(1)

        crack.P1 = z1
        crack.P2 = z2
        crack.length = torch.abs(z2 - z1).item()
        crack.square = [
            torch.minimum(z1.real, z2.real),
            torch.maximum(z1.real, z2.real),
            torch.minimum(z1.imag, z2.imag),
            torch.maximum(z1.imag, z2.imag),
        ]

        for crack in self.cracks:
            for tip in crack.tips:
                tip.domains = self.dd_partition(tip.coords).squeeze(-1).nonzero().item()
                tip.branch_cut_rotation = 0 * tip.branch_cut_rotation[0]

        if len(crack.tips) == 1:
            crack.rice = {
                "is_internal": False,
                "coords": crack.tips[0].coords,
                "angle": crack.tips[0].angle,
                "domain": crack.tips[0].domains,
            }
        elif len(crack.tips) == 2:
            crack.rice = {
                "is_internal": True,
                "coords": (crack.tips[0].coords + crack.tips[1].coords) / 2,
                "angle": crack.tips[0].angle,
                "domain": crack.tips[0].domains,
            }

        for crack in self.cracks:
            d = crack.rice["domain"]
            self.has_crack[d] = 1
            self.crack_is_internal[d] = crack.rice["is_internal"]
            self.crack_coords[d] = crack.rice["coords"]
            self.crack_angle[d] = crack.rice["angle"]
            self.crack_a[d] = crack.length / 2

    def forward(self, z, flat_output=True, real_output=False, force_williams=False):

        if len(z.shape) == 1:
            domains = self.dd_partition(z)
            z_dd = self.unflatten(z, domains)
        else:
            z_dd = z

        if self.enrichment == "williams":
            phi = super(enriched_PIHNN_devo, self).forward(z_dd)
            phi = phi + self.apply_williams(z_dd)
        elif self.enrichment == "rice":
            phi = self.apply_rice(z_dd)

            if force_williams:
                phi = phi[:2] + self.apply_williams(z_dd)

        if flat_output and "domains" in locals():
            phi = self.flatten(phi, domains)

        if real_output:
            return self.apply_real_transformation(z, phi)
        else:
            return phi

    def apply_williams(self, z):

        phi = torch.zeros((2,) + z.shape, device=device) * 1j
        i = 0
        for crack in self.cracks:
            for t in crack.tips:
                d = t.domains
                bcr = t.branch_cut_rotation
                c1 = torch.conj(self.sif[i])
                c2 = self.sif[i] - torch.conj(self.sif[i]) / 2
                rot_sqrt = torch.sqrt(
                    torch.exp(-1j * (bcr + t.angle)) * (z[d] - t.coords)
                )
                phi_williams = c1 * torch.exp(1j * (t.angle + bcr / 2)) * rot_sqrt
                psi_williams = (
                    c2 * torch.exp(1j * (-t.angle + bcr / 2)) * rot_sqrt
                    - 0.5
                    * c1
                    * torch.conj(t.coords)
                    * torch.exp(-1j * bcr / 2)
                    / rot_sqrt
                )
                phi[0, d, :] = phi[0, d, :] + phi_williams / math.sqrt(2 * math.pi)
                phi[1, d, :] = phi[1, d, :] + psi_williams / math.sqrt(2 * math.pi)
                i += 1
        return phi

    def apply_rice(self, z):

        t = torch.exp(-1j * self.crack_angle) * (z - self.crack_coords)
        phi = super(enriched_PIHNN_devo, self).forward(t)  # phi = [f,g]
        phi_h = torch.conj(
            super(enriched_PIHNN_devo, self).forward(torch.conj(t))
        )  # phi_h = [\check{f},\check{g}]

        sigma = (
            torch.sqrt(t) * (~self.crack_is_internal)
            + torch.sqrt(t - self.crack_a)
            * torch.sqrt(t + self.crack_a)
            * self.crack_is_internal
        )

        if self.PDE == "km":
            varphi_0 = (phi[0] * sigma + phi[1]) * self.has_crack + phi[0] * (
                ~self.has_crack
            )
            omega_0 = (phi_h[0] * sigma - phi_h[1]) * self.has_crack + phi[1] * (
                ~self.has_crack
            )
            varphi_t_0 = derivative(varphi_0, t, holom=True)
            psi_0 = omega_0 - t * varphi_t_0
            # Roto-translations
            varphi = torch.exp(1j * self.crack_angle) * varphi_0
            psi = (
                torch.exp(-1j * self.crack_angle) * psi_0
                - torch.conj(self.crack_coords) * varphi_t_0
            )
            varphi_z = varphi_t_0
            return torch.stack(
                [varphi, psi, varphi_z], 0
            )  # We keep varphi_z in order to save some computation time

        elif self.PDE == "km-so":
            varphi_t_0 = (phi[0] / sigma + phi[1]) * self.has_crack + phi[0] * (
                ~self.has_crack
            )
            omega_t_0 = (phi_h[0] / sigma - phi_h[1]) * self.has_crack + phi[1] * (
                ~self.has_crack
            )
            varphi_tt_0 = derivative(varphi_t_0, t, holom=True)
            psi_t_0 = omega_t_0 - t * varphi_tt_0 - varphi_t_0
            # Roto-translations
            varphi_z = varphi_t_0
            psi_z = (
                torch.exp(-2j * self.crack_angle) * psi_t_0
                - torch.conj(self.crack_coords)
                * torch.exp(-1j * self.crack_angle)
                * varphi_tt_0
            )
            varphi_zz = torch.exp(-1j * self.crack_angle) * varphi_tt_0
            return torch.stack(
                [varphi_z, psi_z, varphi_zz], 0
            )  # We keep varphi_zz in order to save some computation time
