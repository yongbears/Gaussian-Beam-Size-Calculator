import numpy as np
import pandas as pd
from ray_transfer import Optics
from Gaussian_propagation import BeamParameters, Propagation

def seed():
    wavelength_1 = 800e-6   # mm
    w_BS = 2.6697           # mm
    
    # initial q-parameter (800 nm)
    z_R_init = (np.pi * w_BS**2) / wavelength_1
    q_init = 0 + 1j * z_R_init
    
    beam1 = BeamParameters(wavelength_1, q_init)
    sim1 = Propagation(beam1)

    # (1) BS to OAP (200 mm)
    sim1.add_element(Optics.propagation_homogeneous(200), "BS to OAP\n(200 mm)", physical_length=200)
    
    # (2) OAP (f=203.2 mm, focusing)
    sim1.add_element(Optics.thin_lens(203.2), "OAP (f=203.2 mm)", physical_length=0)
    
    # (3) OAP to CaF2 (200 mm)
    sim1.add_element(Optics.propagation_homogeneous(200), "OAP to CaF2\n(200 mm)", physical_length=190)
    
    # (4) CaF2 Crystal (2 mm, n=1.43)
    sim1.add_element(Optics.propagation_homogeneous(2.0, n=1.43), "CaF2 (2 mm)", physical_length=2.0)
    
    # run simulation 1 (800 nm)
    df1 = sim1.simulate(step_size_mm=0.01)
    
    # simulation 2 (WLG, central wavelength = 600 nm)
    wavelength_2 = 600e-6

    # update q-parameter for 600 nm beam @ the end of CaF2
    last_row = df1.iloc[-1]; w_at_CaF2 = last_row['w']; R_at_CaF2 = last_row['R']; z_offset = last_row['z']
    if R_at_CaF2 == np.inf:
        inv_R = 0
    else:
        inv_R = 1 / R_at_CaF2
        
    inv_q2 = inv_R - 1j * (wavelength_2 / (np.pi * w_at_CaF2**2))
    q_init_2 = 1 / inv_q2
    
    beam2 = BeamParameters(wavelength_2, q_init_2)
    sim2 = Propagation(beam2)
    
    # (5) CaF2 to PCX (60 mm)
    sim2.add_element(Optics.propagation_homogeneous(60), "CaF2 to PCX\n(60 mm)", physical_length=60)
    
    # (6) PCX (f=60 mm, collimation)
    sim2.add_element(Optics.thin_lens(60), "PCX (f=60 mm)", physical_length=0)
    
    # (7) PCX to SM (160 mm)
    sim2.add_element(Optics.propagation_homogeneous(160), "PCX to SM\n(160 mm)", physical_length=160)
    
    # (8) SM (f=750 mm)
    sim2.add_element(Optics.thin_lens(750), "SM (f=750 mm)", physical_length=0)
    
    # (9) SM to NOPA BBO (760 mm)
    sim2.add_element(Optics.propagation_homogeneous(760), "SM to NOPA\n(760 mm)", physical_length=760)
    
    # run simulation 2
    df2 = sim2.simulate(step_size_mm=0.01)
    
    # combine results
    df2['z'] = df2['z'] + z_offset
    df_total = pd.concat([df1, df2], ignore_index=True)
    
    # results
    print(f"===================== Simulation Results (Seed) ====================")
    summary = df_total[~df_total['label'].isin(['Start', 'prop'])].copy()
    print(summary[['z', 'w', 'w0', 'R', 'label']])
    
    # at NOPA BBO
    final_w = df_total.iloc[-1]['w'] * 1e3
    final_w0 = df_total.iloc[-1]['w0'] * 1e3
    
    print(f"\n at NOPA BBO (seed)")
    print(f" - Beam Size (w) : {final_w:.2f} um")
    print(f" - Beam Waist (w0): {final_w0:.2f} um")
    
    # plot
    sim1.plot(df_total, title="NOPA Seed Beam Path")