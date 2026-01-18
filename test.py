import numpy as np
import matplotlib.pyplot as plt

def waist_necessity():
    wavelength = 400e-6      # [mm]
    distance_z = 203.2       # focus to OAP = OAP focal length [mm]
    target_w_at_OAP = 1.96   # collimated beam size at OAP [mm]

    # w0 = 1 to 100, 1000 step
    w0_candidates_um = np.linspace(1, 100, 1000) 
    w0_candidates_mm = w0_candidates_um * 1e-3
    w_at_OAP_results = []
    
    for w0 in w0_candidates_mm:
        z_R = (np.pi * w0**2) / wavelength
        w_z = w0 * np.sqrt(1 + (distance_z / z_R)**2)
        w_at_OAP_results.append(w_z)
        
    w_at_OAP_results = np.array(w_at_OAP_results)

    # target waist
    idx = np.argmin(np.abs(w_at_OAP_results - target_w_at_OAP))
    found_w0_um = w0_candidates_um[idx]
    found_w_OAP = w_at_OAP_results[idx]
    
    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(w0_candidates_um, w_at_OAP_results, 'b-', linewidth=1.5)
    
    # 목표 지점 표시
    plt.axhline(y=target_w_at_OAP, color='r', linestyle='--', alpha=0.5, label=f'Target Size at OAP ({target_w_at_OAP} mm)')
    plt.axvline(x=found_w0_um, color='g', linestyle='--', alpha=0.5, label=f'Required Waist ({found_w0_um:.1f} um)')
    
    # 포인트 찍기
    plt.plot(found_w0_um, found_w_OAP, 'ro', markersize=10)
    plt.xlabel("Beam waist ($w_0$) [$\mu m$]", fontsize=12)
    plt.ylabel("Collimated beam radius at OAP [mm]", fontsize=12)
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    waist_necessity()