import numpy as np
import matplotlib.pyplot as plt

energy_group = 2 # number of the total Energy group
num_object = 3 # number of objects [Water, Fuel, Water]

object_length = np.array([[30.],[60.], [30.]])
object_mesh = np.empty_like(object_length)
for i in range(num_object):
    object_mesh[i, 0] = 4. * object_length[i, 0]
object_border = np.empty(num_object)
for i in range(num_object):
    if i == 0:
        object_border[i] = object_mesh[i]
    else:
        object_border[i] = object_border[i-1] + object_mesh[i]
num_trial_in = 12000
num_trial_ex = 500
num_mesh_list = [int(object_mesh[i]) for i in range(num_object)] 
num_mesh = sum(num_mesh_list)
###################################################################################################
dif_coeff_input = np.array([[1.41, 1.58, 1.41], [0.117, 0.271, 0.117]])   # 行: エネルギー群数, 列: 媒質の種類
sigma_abs_input = np.array([[0., 0.0032, 0.], [0.0191, 0.093, 0.0191]])   # [[第一群_Water, 第一群_Fuel, 第一群_Water], [第二群_Water,  ...], ...]
sigma_sca_input = np.array([[0.0476, 0.0178, 0.0476], [0., 0., 0.]])
sigma_fis_input = np.array([[0., 0., 0.], [0., 0.168, 0.]])
chi_input = np.array([[1., 1., 1.], [0., 0., 0.]])
###################################################################################################
dif_coeff = np.empty((energy_group, num_mesh))
sigma_abs = np.empty((energy_group, num_mesh))
sigma_sca = np.empty((energy_group, num_mesh))
sigma_fis = np.empty((energy_group, num_mesh))
chi = np.empty((energy_group, num_mesh))
Neutron_flux = np.zeros((energy_group, num_mesh)) # 行: エネルギー群数, 列: 座標
Neutron_flux_past = np.zeros((energy_group, num_mesh))
delta_x_object = np.array([object_length[i] / num_mesh_list[i] for i in range(num_object)])
delta_x = np.empty(num_mesh)
x = np.empty(num_mesh)
#######################################
for g in range(energy_group):
    for i in range(num_mesh):
        if i < object_border[0]:
            dif_coeff[g, i] = dif_coeff_input[g, 0]
            sigma_abs[g, i] = sigma_abs_input[g, 0]
            sigma_fis[g, i] = sigma_fis_input[g, 0]
            sigma_sca[g, i] = sigma_sca_input[g, 0]
            chi[g, i] = chi_input[g, 0]
        elif i < object_border[1]:
            dif_coeff[g, i] = dif_coeff_input[g, 1]
            sigma_abs[g, i] = sigma_abs_input[g, 1]
            sigma_fis[g, i] = sigma_fis_input[g, 1]
            sigma_sca[g, i] = sigma_sca_input[g, 1]
            chi[g, i] = chi_input[g, 1]
        else:
            dif_coeff[g, i] = dif_coeff_input[g, 2]
            sigma_abs[g, i] = sigma_abs_input[g, 2]
            sigma_fis[g, i] = sigma_fis_input[g, 2]
            sigma_sca[g, i] = sigma_sca_input[g, 2]
            chi[g, i] = chi_input[g, 2]
for i in range(num_mesh):
    if i < object_border[0]:
        delta_x[i] = delta_x_object[0]
    elif i < object_border[1]:
        delta_x[i] = delta_x_object[1]
    else:
        delta_x[i] = delta_x_object[2]
for i in range(num_mesh):
    if i == 0:
        x[i] = delta_x[i] / 2.
    else:
        x[i] = x[i-1] + delta_x[i-1] /2. + delta_x[i] / 2.

################################################################################
################################################################################
# main program for the Gauss-Seidel method
sigma_rem = np.array([[sigma_abs[g, i] + sigma_sca[g, i] if g == 0 else sigma_abs[g, i] for i in range(num_mesh)] for g in range(energy_group) ] )
c = np.array([[2.*dif_coeff[g, i]/delta_x[i] if i == num_mesh-1 else 2. * dif_coeff[g, i] * dif_coeff[g, i+1] / (delta_x[i]*dif_coeff[g, i] + delta_x[i]*dif_coeff[g, i+1]) \
               for i in range(num_mesh)] for g in range(energy_group) ] )
const = np.array([[c[g, 0] + (2.*dif_coeff[g, 0])/delta_x[0] + sigma_rem[g, i]*delta_x[i] if i == 0 else c[g, i] + c[g, i-1] +sigma_rem[g, i]*delta_x[i] \
                  for i in range(num_mesh)] for g in range(energy_group) ] )
f = np.array([1.0 / 30.0 if i >= object_border[0] and i < object_border[0] + 120 else 0.0 for i in range(num_mesh)])
f_before = f.copy()
k_eff = 1.0
f_total = 0.
for i in range(num_mesh):
    f_total += f[i] * delta_x[i]
print(object_border)
print("initial condition: (f_total, k_eff, f_total/k_eff) = ", "(", f_total, k_eff, f_total/k_eff, ")")
trial_ex = 1
trial_in = 1
epsilon_ex_k_eff = 1.
epsilon_ex_f = np.empty(num_mesh)
fis_source = np.empty((energy_group, num_mesh))
fis_source_original = np.zeros_like(fis_source)

print(x.shape, delta_x.shape, np.average(delta_x))
print(f.shape, sigma_abs.shape)

#外部反復開始
for ex in range(num_trial_ex):
    # χF/keff の更新
    for g in range(energy_group):
        for i in range(num_mesh):
            fis_source[g, i] = chi[g, i] * f[i] / k_eff
    fis_source_original = fis_source.copy()
    # 内部反復(Φ(group, mesh))
    for g in range(energy_group):
        if g == 1:
            for j in range(num_mesh):
                fis_source[1, j] = fis_source_original[1, j] + sigma_sca[0, j] * Neutron_flux[0, j]
        
        for k in range(num_trial_in):     
            for j in range(num_mesh):
                if j == 0 : # n = 1
                    Neutron_flux[g, j] = (c[g, j]*Neutron_flux[g, j+1] + fis_source[g, j]*delta_x[j]) / const[g, j]
                elif j == num_mesh-1: # n = N
                    Neutron_flux[g, j] = (c[g, j-1]*Neutron_flux[g, j-1] + fis_source[g, j]*delta_x[j]) / const[g, j]
                else: # 2<=n<=N-1
                    Neutron_flux[g, j] = (c[g, j]*Neutron_flux[g, j+1] + c[g, j-1]*Neutron_flux[g, j-1] + fis_source[g, j]*delta_x[j]) / const[g, j]
            epsilon_in =np.array(0)
            epsilon_in = np.array([1. if Neutron_flux_past[g, i] == 0. else abs((Neutron_flux[g, i] - Neutron_flux_past[g, i]) / Neutron_flux_past[g, i]) for i in range(num_mesh) ] )
            if np.max(epsilon_in) < 10. ** (-6.):
                if trial_ex % 100 == 0:
                    print("内部反復は", g+1, "群は", trial_in, "回で収束しました。" "\n")
                break
            if k == num_trial_in - 1:
                print("内部反復は", g+1, "群は", trial_in, "回では収束しませんでした。\n")
            else:
                if trial_in % 10000 == 0:
                    print("内部反復の現在の試行回数: ", trial_in)       
                trial_in += 1
                Neutron_flux_past = Neutron_flux.copy() 
            
    # 内部反復終了
    # 外部反復 誤差判定        
    f_before = f.copy()
  #  f_total_berore = f_total
    f_total = 0.
    
    for i in range(num_mesh):
        f[i] = sigma_fis[0, i]*Neutron_flux[0, i] + sigma_fis[1, i]*Neutron_flux[1, i]
        f_total += f[i] * delta_x[i]

        if f_before[i] == 0.:
            epsilon_ex_f[i] = 0
        else:
            epsilon_ex_f[i] = abs((f[i] - f_before[i]) / f_before[i])

    k_eff_before = k_eff
 #   k_eff = f_total/(f_total_berore/k_eff_before)
    k_eff = f_total
    epsilon_ex_k_eff = abs((k_eff - k_eff_before) / k_eff_before)
    if np.max(epsilon_ex_f) < 10.**(-5.) and epsilon_ex_k_eff < 10. ** (-5.):
        print("外部反復が終了しました")
        print("総内部反復数: ", trial_in)
        print("総外部反復数: ", trial_ex)
        break
    if ex == num_trial_ex - 1:
        print("外部反復は", g+1, "群は", trial_ex, "回では収束しませんでした。\n")
    else:
        if trial_ex % 10 == 0:
            print("外部反復の現在の試行回数: ", trial_ex)
        trial_ex += 1

print("k_eff = ", k_eff,)    
print("Calculation is finished.")
#########################################################################################################
#########################################################################################################
# グラフ作成
y1 = plt.subplot(2, 1, 1)
y1.scatter(x, Neutron_flux[0, :], color ="blue", label = "Energy group: 1")
y1.scatter(x, Neutron_flux[1, :], color ="red", label = "Energy group: 2")
y1.set_title("8_1  外部反復: 29回  総内部反復: 105849回", fontsize = 14, fontname =  "MS Gothic")
y1.set_xlabel("x", fontsize = 14)
y1.set_ylabel("Neutron Flux", fontsize = 14, color ="black")
y1.legend(bbox_to_anchor = (1, 1), loc = "upper right", borderaxespad = 0, fontsize = 14)
plt.ylim(ymin = 0., ymax = 1.0)

y2 = plt.subplot(2, 1, 2)
y2.scatter(x, f, color = "black")
y2.set_ylabel("F(x)", fontsize = 14, color = "black")
y2.set_xlabel("x", fontsize = 14, color = "black")

plt.ylim(ymin = 0., ymax = 0.03)
y1.grid(which = "major", axis = "y", color = "black", alpha = 0.1)
y2.grid(which = "major", axis = "y", color = "black", alpha = 0.1)
plt.show()