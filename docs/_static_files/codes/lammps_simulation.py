#!/usr/bin/env python3
"""
LAMMPS Lennard-Jones fluid simulation with visualization
Generates input scripts, runs LAMMPS, and creates trajectory animations
and thermodynamic property plots.
"""
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import Axes3D


def create_lammps_input():
    """Generate LAMMPS input script for LJ fluid in NPT ensemble."""
    script = """# LJ fluid simulation - NPT ensemble
units lj
atom_style atomic
dimension 3
boundary p p p

# Create FCC lattice (864 atoms)
lattice fcc 0.8442
region box block 0 6 0 6 0 6
create_box 1 box
create_atoms 1 box
mass 1 1.0

# LJ potential
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0

# Neighbor list
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes

# Increase ghost atom communication for RDF
comm_modify cutoff 6.0

# Initial velocities (T* = 2.0)
velocity all create 2.0 482901 dist gaussian

# Thermodynamic output
thermo_style custom step temp pe ke etotal press vol density
thermo 100

# Trajectory output
dump 1 all custom 50 trajectory.lammpstrj id type x y z
dump_modify 1 sort id

# RDF calculation
compute rdf all rdf 100 1 1 cutoff 5.0
fix 3 all ave/time 100 10 1000 c_rdf[*] file rdf.dat mode vector

# MSD calculation
compute msd all msd
fix 4 all ave/time 1 1 1 c_msd[4] file msd.dat

# NPT equilibration (cool from T* = 2.0 to T* = 1.0)
fix 1 all npt temp 2.0 1.0 0.1 iso 1.0 1.0 1.0
timestep 0.005
run 5000

# NPT production (T* = 1.0, P* = 1.0)
unfix 1
fix 2 all npt temp 1.0 1.0 0.1 iso 1.0 1.0 1.0
run 10000

write_data lj_final.data
"""
    with open('lj_fluid.in', 'w') as f:
        f.write(script)
    print('Created lj_fluid.in')


def run_lammps():
    """Run LAMMPS simulation."""
    if not os.path.exists('lj_fluid.in'):
        create_lammps_input()
    
    print('Running LAMMPS simulation...')
    print('(Loading LAMMPS module and running simulation)')
    
    try:
        # Load module and run LAMMPS
        cmd = 'module load lammps/23Jun2022/gpu && lmp -in lj_fluid.in'
        result = subprocess.run(cmd, shell=True, executable='/bin/bash',
                              capture_output=True, text=True, timeout=300)
        
        with open('lj_fluid.log', 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write('\n=== STDERR ===\n')
                f.write(result.stderr)
        
        if result.returncode == 0:
            print('Simulation completed successfully')
            return True
        else:
            print(f'Error running LAMMPS (return code {result.returncode})')
            print(f'Check lj_fluid.log for details')
            return False
    except FileNotFoundError:
        print('ERROR: LAMMPS executable "lmp" not found in PATH')
        print('Please install LAMMPS or modify the executable name')
        return False
    except subprocess.TimeoutExpired:
        print('Simulation timed out after 5 minutes')
        return False


def read_lammpstrj(filename):
    """
    Parse LAMMPS trajectory file.
    
    Returns list of frames, each containing:
    - timestep: simulation step
    - coords: Nx3 array of particle coordinates
    - box: simulation box bounds
    """
    frames = []
    if not os.path.exists(filename):
        print(f'Trajectory file {filename} not found')
        return frames
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        if 'ITEM: TIMESTEP' in lines[i]:
            timestep = int(lines[i+1].strip())
            i += 2
            
            # Number of atoms
            assert 'ITEM: NUMBER OF ATOMS' in lines[i]
            natoms = int(lines[i+1].strip())
            i += 2
            
            # Box bounds
            assert 'ITEM: BOX BOUNDS' in lines[i]
            xlo_xhi = list(map(float, lines[i+1].split()))
            ylo_yhi = list(map(float, lines[i+2].split()))
            zlo_zhi = list(map(float, lines[i+3].split()))
            box = [xlo_xhi, ylo_yhi, zlo_zhi]
            i += 4
            
            # Atom data
            assert 'ITEM: ATOMS' in lines[i]
            i += 1
            coords = []
            for j in range(natoms):
                data = lines[i+j].split()
                # Assumes format: id type x y z
                coords.append([float(data[2]), float(data[3]), float(data[4])])
            i += natoms
            
            frames.append({
                'timestep': timestep, 
                'coords': np.array(coords), 
                'box': box
            })
        else:
            i += 1
    
    print(f'Read {len(frames)} frames from {filename}')
    return frames


def create_animation(frames, output_file='lammps_trajectory.gif', skip=5, max_frames=50):
    """
    Create animated GIF from trajectory frames.
    
    Parameters:
    - frames: list of trajectory frames from read_lammpstrj()
    - output_file: output filename for GIF
    - skip: frame skip factor (1 = all frames, 5 = every 5th frame)
    - max_frames: maximum number of frames to render
    """
    if not frames:
        print('No frames to animate')
        return
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    writer = PillowWriter(fps=10)
    
    print(f'Creating animation with {min(len(frames[::skip]), max_frames)} frames...')
    
    with writer.saving(fig, output_file, dpi=100):
        for idx, frame in enumerate(frames[::skip]):
            if idx >= max_frames:
                break
            
            ax.clear()
            coords = frame['coords']
            box = frame['box']
            
            # Plot atoms as spheres
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                      c='blue', marker='o', s=50, alpha=0.6, 
                      edgecolors='darkblue', linewidths=0.5)
            
            # Set axis limits from simulation box
            ax.set_xlim([box[0][0], box[0][1]])
            ax.set_ylim([box[1][0], box[1][1]])
            ax.set_zlim([box[2][0], box[2][1]])
            
            ax.set_xlabel('x* (σ)', fontsize=12)
            ax.set_ylabel('y* (σ)', fontsize=12)
            ax.set_zlabel('z* (σ)', fontsize=12)
            ax.set_title(f'LJ Fluid: Timestep {frame["timestep"]}', fontsize=14, pad=10)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            writer.grab_frame()
    
    print(f'Animation saved to {output_file}')


def plot_thermodynamics(logfile='lj_fluid.log'):
    """
    Plot thermodynamic quantities from LAMMPS log file.
    
    Extracts and plots:
    - Temperature vs time
    - Pressure vs time
    - Total energy vs time
    - Density vs time
    """
    if not os.path.exists(logfile):
        print(f'Log file {logfile} not found')
        return
    
    step, temp, pe, ke, etotal, press, vol, density = [], [], [], [], [], [], [], []
    
    with open(logfile, 'r') as f:
        reading = False
        for line in f:
            # Look for thermo output header
            if 'Step' in line and 'Temp' in line and 'PotEng' in line:
                reading = True
                continue
            
            if reading:
                # Stop at end of run
                if line.startswith('Loop time') or line.strip() == '':
                    reading = False
                    continue
                
                try:
                    data = list(map(float, line.split()))
                    if len(data) == 8:
                        step.append(data[0])
                        temp.append(data[1])
                        pe.append(data[2])
                        ke.append(data[3])
                        etotal.append(data[4])
                        press.append(data[5])
                        vol.append(data[6])
                        density.append(data[7])
                except:
                    pass
    
    if not step:
        print('No thermodynamic data found in log file')
        return
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Temperature
    axes[0, 0].plot(step, temp, 'b-', linewidth=1.5, alpha=0.8)
    axes[0, 0].set_xlabel('Step', fontsize=12)
    axes[0, 0].set_ylabel('Temperature T*', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(1.0, color='r', linestyle='--', linewidth=2, label='Target T*=1.0')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].set_title('Temperature Evolution', fontsize=13, fontweight='bold')
    
    # Pressure
    axes[0, 1].plot(step, press, 'g-', linewidth=1.5, alpha=0.8)
    axes[0, 1].set_xlabel('Step', fontsize=12)
    axes[0, 1].set_ylabel('Pressure P*', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(1.0, color='r', linestyle='--', linewidth=2, label='Target P*=1.0')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].set_title('Pressure Evolution', fontsize=13, fontweight='bold')
    
    # Total energy
    axes[1, 0].plot(step, etotal, 'purple', linewidth=1.5, alpha=0.8)
    axes[1, 0].set_xlabel('Step', fontsize=12)
    axes[1, 0].set_ylabel('Total Energy E*', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title('Energy Evolution', fontsize=13, fontweight='bold')
    
    # Density
    axes[1, 1].plot(step, density, 'orange', linewidth=1.5, alpha=0.8)
    axes[1, 1].set_xlabel('Step', fontsize=12)
    axes[1, 1].set_ylabel('Density ρ*', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title('Density Evolution', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('lammps_thermodynamics.png', dpi=300, bbox_inches='tight')
    print('Thermodynamics plot saved to lammps_thermodynamics.png')
    
    # Calculate equilibrium averages (second half of simulation)
    eq_idx = len(step) // 2
    avg_temp = np.mean(temp[eq_idx:])
    avg_press = np.mean(press[eq_idx:])
    avg_density = np.mean(density[eq_idx:])
    avg_energy = np.mean(etotal[eq_idx:])
    
    # Standard deviations
    std_temp = np.std(temp[eq_idx:])
    std_press = np.std(press[eq_idx:])
    
    print('\n' + '='*60)
    print('Equilibrated Properties (last half of simulation):')
    print('='*60)
    print(f'  <T*>   = {avg_temp:.4f} ± {std_temp:.4f}')
    print(f'  <P*>   = {avg_press:.4f} ± {std_press:.4f}')
    print(f'  <ρ*>   = {avg_density:.4f}')
    print(f'  <E*>   = {avg_energy:.4f}')
    print('='*60)


def plot_rdf(rdffile='rdf.dat'):
    """
    Plot radial distribution function g(r).
    
    The RDF shows the probability of finding a particle at distance r
    from a reference particle, normalized by the ideal gas density.
    """
    if not os.path.exists(rdffile):
        print(f'RDF file {rdffile} not found')
        return
    
    # Load RDF data (skip header lines)
    data = []
    with open(rdffile, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    # Skip the "timestep num_rows" line
                    if len(parts) == 2 and int(parts[0]) > 100:
                        continue
                    data.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except:
                    pass
    
    data = np.array(data)
    r = data[:, 1]
    g_r = data[:, 2]
    
    # Filter out points where r is too small (removes the diagonal artifact)
    mask = r > 0.2  # Only include data where r* > 0.2
    r = r[mask]
    g_r = g_r[mask]
    
    plt.figure(figsize=(10, 6))
    plt.plot(r, g_r, 'b-', linewidth=2, label='g(r*)')
    plt.xlabel('r* (σ)', fontsize=14)
    plt.ylabel('g(r*)', fontsize=14)
    plt.title('Radial Distribution Function - LJ Fluid (T*=1.0, P*=1.0)', 
              fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 5])
    plt.ylim([0, max(g_r) * 1.1])
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Ideal gas')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('lammps_rdf.png', dpi=300, bbox_inches='tight')
    print('RDF plot saved to lammps_rdf.png')
    
    # Analyze structure
    first_peak_idx = np.argmax(g_r[:50])
    first_peak_pos = r[first_peak_idx]
    first_peak_height = g_r[first_peak_idx]
    
    print(f'\nStructural Analysis:')
    print(f'  First peak position: r* = {first_peak_pos:.3f}')
    print(f'  First peak height:   g(r*) = {first_peak_height:.3f}')
    print(f'  (Expected: r* ≈ 1.1, typical liquid structure)')


def plot_msd(msdfile='msd.dat'):
    """
    Plot mean-square displacement for diffusion analysis.
    
    MSD ~ 6Dt for long times (Einstein relation)
    """
    if not os.path.exists(msdfile):
        print(f'MSD file {msdfile} not found')
        return
    
    data = np.loadtxt(msdfile, comments='#')
    if len(data.shape) == 1:
        # Single column - just MSD values
        time = np.arange(len(data)) * 0.005
        msd = data
    elif data.shape[1] == 2:
        # Two columns: timestep, MSD
        time = data[:, 0] * 0.005  # timestep * dt
        msd = data[:, 1]
    else:
        print(f'Unexpected MSD file format: {data.shape}')
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, msd, 'r-', linewidth=2)
    plt.xlabel('Time t* (τ)', fontsize=14)
    plt.ylabel('MSD (σ²)', fontsize=14)
    plt.title('Mean-Square Displacement', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lammps_msd.png', dpi=300, bbox_inches='tight')
    print('MSD plot saved to lammps_msd.png')
    
    # Estimate diffusion coefficient from linear regime
    if len(time) > 100:
        # Use second half for linear fit
        fit_idx = len(time) // 2
        coeffs = np.polyfit(time[fit_idx:], msd[fit_idx:], 1)
        D = coeffs[0] / 6.0  # Einstein relation: MSD = 6Dt
        
        print(f'\nDiffusion Analysis:')
        print(f'  Diffusion coefficient: D* = {D:.4f} (σ²/τ)')
        print(f'  (Fitted from linear regime: t* > {time[fit_idx]:.2f})')


def main():
    """Main execution function."""
    print('='*60)
    print('  LAMMPS Lennard-Jones Fluid Simulation')
    print('='*60)
    print('\nThis script will:')
    print('  1. Create LAMMPS input file')
    print('  2. Run MD simulation (NPT ensemble)')
    print('  3. Generate trajectory animation (GIF)')
    print('  4. Plot thermodynamic properties')
    print('  5. Analyze structure (RDF) and dynamics (MSD)')
    print('='*60)
    
    # Create input and run simulation
    create_lammps_input()
    success = run_lammps()
    
    if not success:
        print('\nSimulation failed. Creating example plots from dummy data...')
        # Could add mock data generation here for demonstration
        return
    
    # Generate visualizations
    print('\n' + '='*60)
    print('Generating Visualizations')
    print('='*60)
    
    # Trajectory animation
    frames = read_lammpstrj('trajectory.lammpstrj')
    if frames:
        create_animation(frames, 'lammps_trajectory.gif', skip=5, max_frames=50)
    
    # Thermodynamics
    if os.path.exists('lj_fluid.log'):
        plot_thermodynamics('lj_fluid.log')
    
    # Structure
    plot_rdf('rdf.dat')
    
    # Dynamics
    plot_msd('msd.dat')
    
    print('\n' + '='*60)
    print('All Done!')
    print('='*60)
    print('\nGenerated files:')
    print('  - lj_fluid.in              (LAMMPS input script)')
    print('  - lj_fluid.log             (Simulation log)')
    print('  - trajectory.lammpstrj     (Trajectory file)')
    print('  - lammps_trajectory.gif    (Animation)')
    print('  - lammps_thermodynamics.png (Thermo plots)')
    print('  - lammps_rdf.png           (Structure)')
    print('  - lammps_msd.png           (Diffusion)')
    print('='*60)


if __name__ == '__main__':
    main()
