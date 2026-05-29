"""Test various engineering tricks on top of IMM and gravity_ekf_6d.

Techniques tested:
1. Stereo confidence → R scaling (low conf = high noise)
2. NIS-adaptive R (monitor innovation statistics)
3. Velocity physical clamp (soft speed limit)
4. Pre-update jump check (reject implausible observations)
5. Combined best techniques
"""

import numpy as np
import yaml
import sys
sys.path.insert(0, '.')
from loader import load_dataset, Segment, Frame
from filters.imm_filter import IMMFilter
from filters.gravity_ekf_6d import GravityEKF6D
from filters.base import FilterState


def load_config():
    with open('config.yaml') as f:
        return yaml.safe_load(f)


def compute_prediction_errors(results, timestamps, obs_xyz, gravity_vec, horizons=[0.1, 0.2, 0.3, 0.5]):
    """Compute prediction RMSE at multiple horizons."""
    ts = timestamps
    obs = obs_xyz
    errors = {h: [] for h in horizons}
    
    for horizon in horizons:
        for i in range(len(ts) - 3):
            target_time = ts[i] + horizon
            j = np.searchsorted(ts, target_time)
            if j >= len(ts):
                continue
            actual_dt = ts[j] - ts[i]
            if actual_dt < horizon * 0.5:
                continue
            disp = np.linalg.norm(obs[j] - obs[i])
            if disp < 0.05:
                continue
            # Parabola consistency check
            if j - i > 2:
                mid = (i + j) // 2
                dt1 = ts[mid] - ts[i]
                dt2 = ts[j] - ts[i]
                v0e = (obs[j] - obs[i] - 0.5 * gravity_vec * dt2**2) / dt2
                exp_mid = obs[i] + v0e * dt1 + 0.5 * gravity_vec * dt1**2
                if np.linalg.norm(obs[mid] - exp_mid) > 0.20:
                    continue
            # Tracking gate
            if np.linalg.norm(results[i, :3] - obs[i]) > 1.0:
                continue
            # Predict
            pred = results[i, :3] + results[i, 3:6] * actual_dt + 0.5 * gravity_vec * actual_dt**2
            err = np.linalg.norm(pred - obs[j])
            errors[horizon].append(err)
    
    rmses = {}
    for h in horizons:
        if errors[h]:
            rmses[h] = np.sqrt(np.mean(np.array(errors[h])**2))
        else:
            rmses[h] = float('nan')
    return rmses, {h: len(errors[h]) for h in horizons}


# ============================================================
# Technique 1: Stereo confidence → R scaling
# ============================================================
class StereoConfIMM:
    """IMM with R scaled by stereo confidence."""
    
    def __init__(self, config, conf_scale=5.0):
        params = config['filters']['imm'].copy()
        params['focal'] = config['camera']['focal']
        params['gravity'] = config['physics']['gravity']
        params['gravity_vec'] = config['physics']['gravity_vec']
        self.filt = IMMFilter(**params)
        self.conf_scale = conf_scale  # R multiplier when conf=0
    
    def process_segment(self, frames):
        self.filt.reset()
        n = len(frames)
        results = np.zeros((n, 9))
        
        for i, frame in enumerate(frames):
            if i > 0:
                dt = frame.timestamp - frames[i-1].timestamp
                if dt > 0:
                    self.filt.predict(dt)
            
            # Adjust R based on stereo_conf before update
            conf = max(0.01, frame.stereo_conf)
            r_factor = 1.0 + (self.conf_scale - 1.0) * (1.0 - conf)
            
            # Temporarily scale R_base
            orig_R = self.filt.R_base
            self.filt.R_base *= r_factor
            
            state = self.filt.update(frame.obs_x, frame.obs_y, frame.obs_z)
            
            # Restore R
            self.filt.R_base = orig_R
            
            results[i] = state.as_array()
        
        return results


# ============================================================
# Technique 2: NIS-Adaptive R (monitor innovation squared)
# ============================================================
class NISAdaptiveIMM:
    """IMM with NIS-based R adaptation."""
    
    def __init__(self, config, nis_window=10, target_nis=3.0, adapt_rate=0.1):
        params = config['filters']['imm'].copy()
        params['focal'] = config['camera']['focal']
        params['gravity'] = config['physics']['gravity']
        params['gravity_vec'] = config['physics']['gravity_vec']
        self.filt = IMMFilter(**params)
        self.nis_window = nis_window
        self.target_nis = target_nis  # Expected NIS for 3DOF
        self.adapt_rate = adapt_rate
        self.nis_history = []
        self.r_scale = 1.0
    
    def process_segment(self, frames):
        self.filt.reset()
        self.nis_history = []
        self.r_scale = 1.0
        n = len(frames)
        results = np.zeros((n, 9))
        
        for i, frame in enumerate(frames):
            if i > 0:
                dt = frame.timestamp - frames[i-1].timestamp
                if dt > 0:
                    self.filt.predict(dt)
            
            # Apply adaptive R scale
            orig_R = self.filt.R_base
            self.filt.R_base *= self.r_scale
            
            state = self.filt.update(frame.obs_x, frame.obs_y, frame.obs_z)
            
            # Restore R
            self.filt.R_base = orig_R
            
            # Get NIS from diagnostics
            diag = self.filt.get_diagnostics()
            innov = diag['innovation']
            S = diag['S']
            try:
                nis = innov @ np.linalg.inv(S) @ innov
            except:
                nis = self.target_nis
            
            self.nis_history.append(nis)
            
            # Adapt R scale based on recent NIS
            if len(self.nis_history) >= self.nis_window:
                mean_nis = np.mean(self.nis_history[-self.nis_window:])
                # If NIS too high → increase R (trust obs less)
                # If NIS too low → decrease R (trust obs more)
                ratio = mean_nis / self.target_nis
                self.r_scale *= (1.0 + self.adapt_rate * (ratio - 1.0))
                self.r_scale = np.clip(self.r_scale, 0.1, 10.0)
            
            results[i] = state.as_array()
        
        return results


# ============================================================
# Technique 3: Velocity physical clamp
# ============================================================
class VelClampIMM:
    """IMM with soft velocity clamping after each update."""
    
    def __init__(self, config, max_speed=12.0, clamp_alpha=0.5):
        params = config['filters']['imm'].copy()
        params['focal'] = config['camera']['focal']
        params['gravity'] = config['physics']['gravity']
        params['gravity_vec'] = config['physics']['gravity_vec']
        self.filt = IMMFilter(**params)
        self.max_speed = max_speed
        self.clamp_alpha = clamp_alpha  # 0=no clamp, 1=hard clamp
    
    def process_segment(self, frames):
        self.filt.reset()
        n = len(frames)
        results = np.zeros((n, 9))
        
        for i, frame in enumerate(frames):
            if i > 0:
                dt = frame.timestamp - frames[i-1].timestamp
                if dt > 0:
                    self.filt.predict(dt)
            
            state = self.filt.update(frame.obs_x, frame.obs_y, frame.obs_z)
            vel = state.velocity()
            speed = np.linalg.norm(vel)
            
            if speed > self.max_speed:
                # Soft clamp: blend toward limited speed
                limited = vel * (self.max_speed / speed)
                blended = vel * (1 - self.clamp_alpha) + limited * self.clamp_alpha
                
                # Apply to all model states
                for j in range(self.filt.n_models):
                    self.filt.x[j][3:6] = blended
                
                state = FilterState(
                    x=state.x, y=state.y, z=state.z,
                    vx=blended[0], vy=blended[1], vz=blended[2])
            
            results[i] = state.as_array()
        
        return results


# ============================================================
# Technique 4: Pre-update jump check
# ============================================================
class JumpCheckIMM:
    """IMM with pre-update observation jump rejection."""
    
    def __init__(self, config, max_jump_factor=3.0):
        params = config['filters']['imm'].copy()
        params['focal'] = config['camera']['focal']
        params['gravity'] = config['physics']['gravity']
        params['gravity_vec'] = config['physics']['gravity_vec']
        self.filt = IMMFilter(**params)
        self.max_jump_factor = max_jump_factor
        self.g_vec = np.array(config['physics']['gravity_vec'], dtype=float)
    
    def process_segment(self, frames):
        self.filt.reset()
        n = len(frames)
        results = np.zeros((n, 9))
        last_obs = None
        last_vel = np.zeros(3)
        
        for i, frame in enumerate(frames):
            obs = np.array([frame.obs_x, frame.obs_y, frame.obs_z])
            
            if i > 0:
                dt = frame.timestamp - frames[i-1].timestamp
                if dt > 0:
                    self.filt.predict(dt)
                    
                    # Check if jump is physically plausible
                    if last_obs is not None and dt > 0:
                        expected_pos = last_obs + last_vel * dt + 0.5 * self.g_vec * dt**2
                        jump = np.linalg.norm(obs - expected_pos)
                        max_allowed = self.max_jump_factor * (15.0 * dt + 0.5 * 9.81 * dt**2 + 0.3)
                        
                        if jump > max_allowed:
                            # Skip update, use prediction only
                            results[i] = self.filt.get_state().as_array()
                            continue
            
            state = self.filt.update(frame.obs_x, frame.obs_y, frame.obs_z)
            last_obs = obs.copy()
            last_vel = state.velocity()
            results[i] = state.as_array()
        
        return results


# ============================================================
# Technique 5: Stereo conf + NIS combined
# ============================================================
class CombinedIMM:
    """IMM with stereo conf weighting + NIS monitoring + jump check."""
    
    def __init__(self, config, conf_scale=3.0, nis_window=8, 
                 adapt_rate=0.05, max_jump_factor=4.0):
        params = config['filters']['imm'].copy()
        params['focal'] = config['camera']['focal']
        params['gravity'] = config['physics']['gravity']
        params['gravity_vec'] = config['physics']['gravity_vec']
        self.filt = IMMFilter(**params)
        self.conf_scale = conf_scale
        self.nis_window = nis_window
        self.adapt_rate = adapt_rate
        self.max_jump_factor = max_jump_factor
        self.g_vec = np.array(config['physics']['gravity_vec'], dtype=float)
        self.nis_history = []
        self.r_scale = 1.0
    
    def process_segment(self, frames):
        self.filt.reset()
        self.nis_history = []
        self.r_scale = 1.0
        n = len(frames)
        results = np.zeros((n, 9))
        last_obs = None
        last_vel = np.zeros(3)
        
        for i, frame in enumerate(frames):
            obs = np.array([frame.obs_x, frame.obs_y, frame.obs_z])
            
            if i > 0:
                dt = frame.timestamp - frames[i-1].timestamp
                if dt > 0:
                    self.filt.predict(dt)
                    
                    # Jump check
                    if last_obs is not None and dt > 0:
                        expected_pos = last_obs + last_vel * dt + 0.5 * self.g_vec * dt**2
                        jump = np.linalg.norm(obs - expected_pos)
                        max_allowed = self.max_jump_factor * (15.0 * dt + 0.5 * 9.81 * dt**2 + 0.3)
                        
                        if jump > max_allowed:
                            results[i] = self.filt.get_state().as_array()
                            continue
            
            # Stereo confidence scaling
            conf = max(0.01, frame.stereo_conf)
            conf_factor = 1.0 + (self.conf_scale - 1.0) * (1.0 - conf)
            total_r_factor = conf_factor * self.r_scale
            
            # Apply R scaling
            orig_R = self.filt.R_base
            self.filt.R_base *= total_r_factor
            
            state = self.filt.update(frame.obs_x, frame.obs_y, frame.obs_z)
            
            # Restore R
            self.filt.R_base = orig_R
            
            # NIS adaptation
            diag = self.filt.get_diagnostics()
            innov = diag['innovation']
            S = diag['S']
            try:
                nis = innov @ np.linalg.inv(S) @ innov
            except:
                nis = 3.0
            self.nis_history.append(nis)
            
            if len(self.nis_history) >= self.nis_window:
                mean_nis = np.mean(self.nis_history[-self.nis_window:])
                ratio = mean_nis / 3.0
                self.r_scale *= (1.0 + self.adapt_rate * (ratio - 1.0))
                self.r_scale = np.clip(self.r_scale, 0.2, 5.0)
            
            last_obs = obs.copy()
            last_vel = state.velocity()
            results[i] = state.as_array()
        
        return results


# ============================================================
# Main evaluation
# ============================================================
if __name__ == '__main__':
    config = load_config()
    gravity_vec = np.array(config['physics']['gravity_vec'], dtype=float)
    segs = load_dataset('../data', '1')
    print(f'Testing on {len(segs)} throw segments\n')
    
    # Define techniques to test
    techniques = {
        'IMM (baseline)': lambda: IMMTechnique(config),
        'StereoConf (scale=3)': lambda: StereoConfIMM(config, conf_scale=3.0),
        'StereoConf (scale=5)': lambda: StereoConfIMM(config, conf_scale=5.0),
        'StereoConf (scale=10)': lambda: StereoConfIMM(config, conf_scale=10.0),
        'NIS-Adaptive (rate=0.1)': lambda: NISAdaptiveIMM(config, adapt_rate=0.1),
        'NIS-Adaptive (rate=0.05)': lambda: NISAdaptiveIMM(config, adapt_rate=0.05),
        'VelClamp (12m/s, a=0.5)': lambda: VelClampIMM(config, max_speed=12.0, clamp_alpha=0.5),
        'VelClamp (12m/s, a=1.0)': lambda: VelClampIMM(config, max_speed=12.0, clamp_alpha=1.0),
        'JumpCheck (factor=3)': lambda: JumpCheckIMM(config, max_jump_factor=3.0),
        'JumpCheck (factor=4)': lambda: JumpCheckIMM(config, max_jump_factor=4.0),
        'Combined (best)': lambda: CombinedIMM(config),
    }
    
    # IMM baseline wrapper
    class IMMTechnique:
        def __init__(self, config):
            params = config['filters']['imm'].copy()
            params['focal'] = config['camera']['focal']
            params['gravity'] = config['physics']['gravity']
            params['gravity_vec'] = config['physics']['gravity_vec']
            self.filt = IMMFilter(**params)
        
        def process_segment(self, frames):
            return self.filt.process_segment(frames)
    
    # Run all techniques
    print(f"{'Technique':<28} {'RMSE@0.1':<9} {'RMSE@0.2':<9} {'RMSE@0.3':<9} {'RMSE@0.5':<9} {'N@0.1':<7}")
    print('=' * 80)
    
    for name, factory in techniques.items():
        all_errors = {h: [] for h in [0.1, 0.2, 0.3, 0.5]}
        
        for seg in segs:
            tech = factory()
            results = tech.process_segment(seg.frames)
            ts = seg.timestamps
            obs = seg.obs_xyz
            
            for horizon in [0.1, 0.2, 0.3, 0.5]:
                for i in range(len(ts) - 3):
                    target_time = ts[i] + horizon
                    j = np.searchsorted(ts, target_time)
                    if j >= len(ts):
                        continue
                    actual_dt = ts[j] - ts[i]
                    if actual_dt < horizon * 0.5:
                        continue
                    disp = np.linalg.norm(obs[j] - obs[i])
                    if disp < 0.05:
                        continue
                    if j - i > 2:
                        mid = (i + j) // 2
                        dt1 = ts[mid] - ts[i]
                        dt2 = ts[j] - ts[i]
                        v0e = (obs[j] - obs[i] - 0.5 * gravity_vec * dt2**2) / dt2
                        exp_mid = obs[i] + v0e * dt1 + 0.5 * gravity_vec * dt1**2
                        if np.linalg.norm(obs[mid] - exp_mid) > 0.20:
                            continue
                    if np.linalg.norm(results[i, :3] - obs[i]) > 1.0:
                        continue
                    pred = results[i, :3] + results[i, 3:6] * actual_dt + 0.5 * gravity_vec * actual_dt**2
                    err = np.linalg.norm(pred - obs[j])
                    all_errors[horizon].append(err)
        
        row = f'{name:<28}'
        for h in [0.1, 0.2, 0.3, 0.5]:
            errs = all_errors[h]
            if errs:
                r = np.sqrt(np.mean(np.array(errs)**2))
                row += f' {r:<8.3f}'
            else:
                row += f' ---     '
        row += f' {len(all_errors[0.1]):<7}'
        print(row)
