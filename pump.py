import numpy as np
from ray_transfer import Optics
from Gaussian_propagation import BeamParameters, Propagation

def pump():
    wavelength = 400e-6     # mm
    w_BS = 2.326      # mm
    
    # initial q-parmeter (assume collimation)
    z_R_init = (np.pi * w_BS**2) / wavelength
    q_init = 0 + 1j * z_R_init
    
    beam = BeamParameters(wavelength, q_init)
    sim = Propagation(beam)

    # Optics propagation
    # (1) BS to PCX Lens (50mm)
    sim.add_element(Optics.propagation_homogeneous(50), "BS to PCX\n(50 mm)", physical_length=50)
    
    # (2) PCX Lens (f=300mm)
    sim.add_element(Optics.thin_lens(300), "PCX (f=300 mm)", physical_length=0)
    
    # (3) PCX to BBO (320mm)
    sim.add_element(Optics.propagation_homogeneous(320), "PCX to BBO\n(320 mm)", physical_length=320)
    
    # (4) BBO Crystal (0.15mm, n=1.67)
    sim.add_element(Optics.propagation_homogeneous(0.15, n=1.67), "BBO (0.15 mm)", physical_length=0.15)
    
    # (5) BBO to OAP
    sim.add_element(Optics.propagation_homogeneous(200), "BBO to OAP\n(200 mm)", physical_length=200)
    
    # (6) OAP (f=203.2mm, collimation)
    sim.add_element(Optics.thin_lens(203.2), "OAP (f=203.2 mm)", physical_length=0)
    
    # (7) OAP -> SM (505 mm)
    sim.add_element(Optics.propagation_homogeneous(505), "OAP to SM\n(505 mm)", physical_length=505)
    
    # (8) SM (f=200mm) 
    sim.add_element(Optics.thin_lens(200), "SM (f=200 mm)", physical_length=0)
    
    # (9) SM -> NOPA BBO (220mm)
    sim.add_element(Optics.propagation_homogeneous(220), "SM to NOPA\n(220 mm)", physical_length=220)

    # run_simualation
    df = sim.simulate(step_size_mm=0.01)

    # results
    print(f"===================== Simulation Results (Pump) ====================")
    summary = df[~df['label'].isin(['Start', 'prop'])].copy()
    print(summary[['z', 'w', 'w0', 'R', 'label']])
    
    # at NOPA BBO
    final_w = df.iloc[-1]['w'] * 1e3
    final_w0 = df.iloc[-1]['w0'] * 1e3
    
    print(f"\n at NOPA BBO (pump)")
    print(f" - Beam Size (w) : {final_w:.2f} um")
    print(f" - Beam Waist (w0): {final_w0:.2f} um\n")
    
    # plot beam size
    sim.plot(df, title="NOPA Pump Beam Path")