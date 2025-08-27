"""
Test from Section 4.2 in https://doi.org/10.1016/j.engfracmech.2025.111133
"""
import sys
# sys.path.append(r"C:\Windows\System32\site-packages\python3.10")
sys.path.append(r"C:\Users\Nicolas\Desktop\Stage_2A\Deep Learning\pihnn-main")

import pihnn.crack_detection as cd

import pihnn.nn as nn
import pihnn.utils as utils
import pihnn.geometries as geom
import pihnn.graphics as graphics
import pihnn.bc as bc
import pihnn.crack_finding as cf


import os
import torch
import scipy
import numpy as np

# Network parameters 
n_epochs = 6000 # Number of epochs
learn_rate = 1e-4 # Initial learning rate
scheduler_apply = [1000, 2000, 3200, 4500,5500]
units = [1, 10, 10, 10, 1] # Units in each network layer
np_train = 300 # Number of training points on domain boundary
np_test = 20 # Number of test points on the domain boundary
beta = 0.5 # Initialization parameter
gauss = 3 # Initalization parameter


# -----------------------------------
# Domaine géométrique et conditions aux limites
# -----------------------------------

h = 10  # Demi-hauteur du domaine
l = 10  # Demi-longueur du domaine
n_segments = 50  # Nombre de segments pour chaque ligne

# Contraintes imposées en haut et en bas (traction/compression pure verticale)
sig_ext_t = 1j   # Traction en haut : σ_yy = +1
sig_ext_b = -1j

# # Stockage des lignes
# line_top = []
# line_bottom = []
# line_right=[]
# line_left = []

# # Abscisses et ordonnées des points de découpe
# x_vals = np.linspace(-l, l, n_segments + 1)
# y_vals = np.linspace(-h, h, n_segments + 1)


# midpoints_complex_bottom=[]
# # Ligne inférieure (y = -h) : contrainte σ_yy = -1j
# for i in range(n_segments):
#     P1 = [x_vals[i], -h]
#     P2 = [x_vals[i + 1], -h]
#     seg = geom.line(P1=P1, P2=P2, bc_type=bc.stress_bc(), bc_value=sig_ext_b)
#     line_bottom.append(seg)
        
#     x_mid = 0.5 * (x_vals[i] + x_vals[i + 1])
#     z_mid = np.complex128(x_mid - h * 1j)
#     midpoints_complex_bottom.append(z_mid)
    
line1 = geom.line(P1=[-l, -h], P2=[l, -h], bc_type=bc.stress_bc(), bc_value=sig_ext_b)


# Côté droit (x = +l) : déplacement nul (encastrement vertical et horizontal)

# line2 = geom.line(P1=[l, -h], P2=[l, h], bc_type=bc.displacement_bc(), bc_value=0 + 0j)

line2 = geom.line(P1=[l, -h], P2=[l, h], bc_type=bc.stress_bc(), bc_value=0 + 0j)


# midpoints_complex_right = []

# for i in range(n_segments):
#     y_mid = 0.5 * (y_vals[i] + y_vals[i + 1])
#     z_mid = np.complex128(l + 1j * y_mid)
#     midpoints_complex_right.append(z_mid)
    
    



# Ligne supérieure (y = +h) : contrainte σ_yy = +1j
# midpoints_complex_top=[]
# for i in range(n_segments):
#     P1 = [x_vals[i + 1], h]
#     P2 = [x_vals[i], h]  # ordre inversé pour sens horaire de la frontière
#     seg = geom.line(P1=P1, P2=P2, bc_type=bc.stress_bc(), bc_value=sig_ext_t)
#     line_top.append(seg)

#     x_mid = 0.5 * (x_vals[i] + x_vals[i + 1])
#     z_mid = np.complex128(x_mid + h * 1j)
#     midpoints_complex_top.append(z_mid)
    
line3 = geom.line(P1=[-l, h], P2=[l, h], bc_type=bc.stress_bc(), bc_value=sig_ext_t)


# Côté gauche (x = -l) : contrainte nulle (bord libre)





# midpoints_complex_left = []
# for i in range(n_segments):
#     P1 = [-l, y_vals[i + 1]]  # sens horaire
#     P2 = [-l, y_vals[i]]
#     seg = geom.line(P1=P1, P2=P2, bc_type=bc.stress_bc(), bc_value=0 + 0j)
#     line_left.append(seg)

#     y_mid = 0.5 * (y_vals[i] + y_vals[i + 1])
#     z_mid = np.complex128(-l + 1j * y_mid)
#     midpoints_complex_left.append(z_mid)


line4 = geom.line(P1=[-l, h], P2=[-l, -h], bc_type=bc.stress_bc(), bc_value=0 + 0j)

# ----------------------------------------------------------
# Fissure horizontale au centre du domaine, de -3.5 à +3.5
# ----------------------------------------------------------
crack = geom.line(P1= -3 - 0j, P2= 3 + 0j, bc_type=bc.stress_bc())

# # Définition des extrémités (tips) de fissure
# crack.add_crack_tip(tip_side=0)  # Extrémité gauche
# crack.add_crack_tip(tip_side=1)  # Extrémité droite

# # Construction de la frontière complète avec enrichissement XFEM de type Rice
# # boundary = geom.boundary(line_bottom + [line2] + line_top + [line4] + [crack], np_train, np_test, enrichment='rice')
# boundary = geom.boundary([line1, line2, line3, line4, crack], np_train, np_test, enrichment='rice')

# # Definition of NN
# model = nn.enriched_PIHNN('km', units, boundary)

# if (__name__=='__main__'):

#     model.initialize_weights('exp', beta, boundary.extract_points(10*np_train)[0], gauss)
#     loss_train, loss_test = utils.train(boundary, model, n_epochs, learn_rate, scheduler_apply)
#     graphics.plot_loss(loss_train, loss_test)
#     tria = graphics.get_triangulation(boundary)    
#     graphics.plot_sol(tria, model, apply_crack_bounds=True) # We bound the crack singularities for the plot
    
    # disp_top=cf.print_displacement(model, midpoints_complex_top)
    # disp_bottom=cf.print_displacement(model, midpoints_complex_bottom)
    
    # stress_top=cf.print_stresses(model, midpoints_complex_top)
    # stress_bottom=cf.print_stresses(model, midpoints_complex_bottom)
    # stress_left=cf.print_stresses(model, midpoints_complex_left)
    # stress_right=cf.print_stresses(model, midpoints_complex_right)
    
# print('top', stress_top)  
# print('bottom',stress_bottom) 


# print(disp_top)  
# print(disp_bottom)  
    
########################################################

#collect data


# # Points choisis
# z_data = torch.tensor([
#     -6 + 4j, -2 + 4j, 2 + 4j, 6 + 4j,
#     -6 - 4j, -2 - 4j, 2 - 4j, 6 - 4j
# ], dtype=torch.cfloat).requires_grad_(True)




# # Valeurs attendues (ex : contrainte nulle aux points libres)
# sig_xx_target, sig_yy_target, sig_xy_target, _, _ = model(z_data, real_output=True)



sig_xx_target = torch.tensor([-0.0853,  0.0067,  0.0068, -0.0859, -0.0860,  0.0076,  0.0087, -0.0836])
sig_yy_target = torch.tensor([1.1788, 0.8001, 0.7978, 1.1751, 1.1767, 0.8000, 0.7986, 1.1748])
sig_xy_target = torch.tensor([0.0538, 0.2477, -0.2496, -0.0526, -0.0532, -0.2485, 0.2490, 0.0531])

n_epochs = 2000 # Number of epochs
learn_rate = 1e-5 # Initial learning rate
# scheduler_apply = [1000, 2000, 3500, 5000, 6500, 8000, 9200] # At which epoch to execute scheduler
scheduler_apply = [500,1000,1500]


line1 = geom.line(P1=[-l, -h], P2=[l, -h], bc_type=bc.stress_bc(), bc_value=sig_ext_b)
line2 = geom.line(P1=[l, -h], P2=[l, h], bc_type=bc.stress_bc(), bc_value=0 + 0j)
line3 = geom.line(P1=[-l, h], P2=[l, h], bc_type=bc.stress_bc(), bc_value=sig_ext_t)
line4 = geom.line(P1=[-l, h], P2=[-l, -h], bc_type=bc.stress_bc(), bc_value=0 + 0j)

# # Stockage des lignes
# line_top = []
# line_bottom = []
# line_left = []
# line_right = []

# # Abscisses des points de découpe
# x_vals = np.linspace(-l, l, n_segments + 1)


# # Ligne inférieure (y = -h) : 
# for i in range(n_segments):
#     P1 = [x_vals[i], -h]
#     P2 = [x_vals[i + 1], -h]
    
#     s_xx, s_yy, s_xy = stress_bottom[i]
#     nx, ny = 0.0, -1.0  # Normale vers le bas
#     t_x = s_xx * nx + s_xy * ny
#     t_y = s_xy * nx + s_yy * ny
#     bc_val = complex(t_x, t_y)


#     seg = geom.line(P1=P1, P2=P2, bc_type=bc.stress_bc(), bc_value=bc_val)
#     line_bottom.append(seg)




# # Ligne supérieure (y = +h) : 
# for i in range(n_segments):
#     P1 = [x_vals[i + 1], h]
#     P2 = [x_vals[i], h]  # ordre inversé pour sens horaire de la frontière
    
#     s_xx, s_yy, s_xy = stress_top[i]
#     nx, ny = 0.0, 1.0  # Normale vers le haut
#     t_x = s_xx * nx + s_xy * ny
#     t_y = s_xy * nx + s_yy * ny
#     bc_val = complex(t_x, t_y)
    
#     seg = geom.line(P1=P1, P2=P2, bc_type=bc.stress_bc(), bc_value=bc_val)
#     line_top.append(seg)




# # Côté gauche (x = -l) :



# y_vals = np.linspace(-h, h, n_segments + 1)

# for i in range(n_segments):
#     P1 = [-l, y_vals[i + 1]]  # sens horaire
#     P2 = [-l, y_vals[i]]
    
#     s_xx, s_yy, s_xy = stress_left[i]
#     nx, ny = -1.0, 0.0  # Normale vers la gauche
#     t_x = s_xx * nx + s_xy * ny
#     t_y = s_xy * nx + s_yy * ny
#     bc_val = complex(t_x, t_y)

#     seg = geom.line(P1=P1, P2=P2, bc_type=bc.stress_bc(), bc_value=bc_val)
#     line_left.append(seg)
    

# # Côté droit (x = +l) : )



# for i in range(n_segments):
#     P1 = [l, y_vals[i]]        # sens horaire
#     P2 = [l, y_vals[i + 1]]
    
#     s_xx, s_yy, s_xy = stress_right[i]
#     nx, ny = 1.0, 0.0  # Normale vers la droite
#     t_x = s_xx * nx + s_xy * ny
#     t_y = s_xy * nx + s_yy * ny
#     bc_val = complex(t_x, t_y)
    
#     seg = geom.line(P1=P1, P2=P2, bc_type=bc.stress_bc(), bc_value=bc_val)
#     line_right.append(seg)



# line2 = geom.line(P1=[l, -h], P2=[l, h], bc_type=bc.displacement_bc(), bc_value=0 + 0j)


#crack initial




crack = geom.line(P1= - 3 - 0j, P2= 3 + 0j, bc_type=bc.stress_bc())


crack.add_crack_tip(tip_side=0)  # Extrémité gauche
crack.add_crack_tip(tip_side=1)  # Extrémité droite


# boundary = geom.boundary(line_bottom + line_right + line_top + line_left + [crack], np_train, np_test, enrichment='rice')
boundary = geom.boundary([line1, line2, line3, line4, crack], np_train, np_test, enrichment='rice')



# Definition of NN

model = nn.enriched_PIHNN_finding('km', units, boundary)


if (__name__=='__main__'):
    model.initialize_weights('exp', beta, boundary.extract_points(10*np_train)[0], gauss)
    loss_train, loss_test,ListeZ1 = utils.train_finding(sig_xx_target, sig_yy_target, sig_xy_target, boundary, model, n_epochs, 
                                                        learn_rate, scheduler_apply, scheduler_gamma=0.5)
    graphics.plot_loss(loss_train, loss_test)
    tria = graphics.get_triangulation(boundary)    
    graphics.plot_sol(tria, model, apply_crack_bounds=True)

    




    




