import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pandas as pd
from ray_transfer import ABCDMatrix, Optics

@dataclass
class BeamParameters:
    """ Gaussian beam parameters state container """
    wavelength: float
    q: complex
    
    def __post_init__(self):
        self._update_params()
    
    def _update_params(self):
        """
        Calculates real beam parameters (w, R, w0) from complex q
        """
        if self.q == 0:
            raise ValueError("Invalid q parameter: cannot be zero")

        inv_q = 1 / self.q
        
        # 1. Beam Radius (w)
        imag_part = inv_q.imag
        if imag_part >= 0:
            self.w = np.inf
        else:
            self.w = np.sqrt(-self.wavelength / (np.pi * imag_part))
        
        # 2. Radius of Curvature (R)
        real_part = inv_q.real
        if real_part == 0:
            self.R = np.inf
        else:
            self.R = 1 / real_part
            
        # 3. Rayleigh Range & Waist Size (w0)
        self.z_R = self.q.imag
        self.z_relative = self.q.real
        
        if self.z_R > 0:
            self.w0 = np.sqrt(self.wavelength * self.z_R / np.pi)
        else:
            self.w0 = 0.0
        
    def __repr__(self):
        return (f"BeamParameters(λ={self.wavelength*1e6:.0f}nm, "
                f"w={self.w*1e3:.1f}μm, R={self.R:.1f}mm, w0={self.w0*1e3:.1f}μm)")


class Propagation:
    """
    Propagation of Gaussian beam through optics
    """
    
    def __init__(self, initial_beam: BeamParameters):
        self.initial_beam = initial_beam
        self.elements = []
        
    def add_element(self, matrix: ABCDMatrix, name: str, physical_length: float = 0.0):
        """
        Args:
            matrix: ABCD matrix for optics
            name: label for the element
            physical_length: propagation length IN the optics [mm]
        """
        self.elements.append({
            'matrix': matrix,
            'name': name,
            'length': physical_length
        })
    
    def simulate(self, step_size_mm: float = 0.01) -> pd.DataFrame:
        """
        Args:
            step_size_mm: Step size for spatial slicing in [mm]
        Returns:
            simulation history DataFrame (z, w, R, w0)
        """
        current_q = self.initial_beam.q
        wavelength = self.initial_beam.wavelength
        
        history = []
        current_z = 0.0
        
        b = BeamParameters(wavelength, current_q)
        history.append({'z': current_z, 'w': b.w, 'R': b.R, 'w0': b.w0, 'label': 'Start'})
        
        for elem in self.elements:
            matrix = elem['matrix']; length = elem['length']; name = elem['name']
            
            if length > 0:
                steps = int(np.ceil(length / step_size_mm))
                dz = length / steps 
                
                effective_n = length / matrix.B if matrix.B != 0 else 1.0
                
                for _ in range(steps):
                    step_matrix = Optics.propagation_homogeneous(dz, effective_n)
                    current_q = step_matrix.transform_q(current_q)
                    current_z += dz
                    
                    b = BeamParameters(wavelength, current_q)
                    history.append({
                        'z': current_z, 
                        'w': b.w, 
                        'R': b.R, 
                        'w0': b.w0,
                        'label': 'prop'
                    })
                
                history[-1]['label'] = name

            else:
                current_q = matrix.transform_q(current_q)
                b = BeamParameters(wavelength, current_q)
                history.append({'z': current_z, 'w': b.w, 'R': b.R, 'w0': b.w0,'label': name})
                
        return pd.DataFrame(history)

    def plot(self, df: pd.DataFrame, title: str = "Beam Propagation"):
        plt.figure(figsize=(12, 8))
        
        z_vals = df['z']; w_vals = df['w'] 
        max_w = max(w_vals); max_z = max(z_vals)
        
        # plot beam profile
        plt.plot(z_vals, w_vals, 'b-', linewidth=1.5, label='Beam Radius ($w$)')
        plt.plot(z_vals, -w_vals, 'b-', linewidth=1.5)
        plt.fill_between(z_vals, -w_vals, w_vals, color='blue', alpha=0.05)
        
        # filter optics for labeling
        elements_df = df[~df['label'].isin(['Start', 'prop'])].reset_index(drop=True)
        
        # plot
        base_height = max_w * 1.2
        step_height = max_w * 0.4 
        
        for i, row in elements_df.iterrows():
            z_pos = row['z']
            
            # 빨간 점선 (Optical Element 위치)
            plt.axvline(x=z_pos, color='red', linestyle='--', alpha=0.5)
            
            # 3단 높이 조절 (0, 1, 2, 0, 1, 2...)
            level = i % 4
            text_y = base_height + (level * step_height)
            
            # 텍스트 출력
            # rotation=90: 세로로 회전하여 가로 겹침 방지
            # va='bottom': 텍스트 시작점이 라인 위쪽
            plt.text(z_pos, text_y, row['label'], 
                     #rotation=90, 
                     color='black', 
                     va='bottom', 
                     ha='center', 
                     fontsize=9,
                     backgroundcolor='white') # 텍스트 뒤에 흰 배경을 둬서 가독성 확보

        plt.xlabel('Propagation Distance (mm)', fontsize=12)
        plt.ylabel('Beam Radius (mm)', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # 텍스트가 잘리지 않도록 Y축 상단 여유 공간 확보 (중요!)
        plt.ylim(-max_w * 1.5, max_w * 3.5) 
        
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

        