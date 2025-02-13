import numpy as np
import gc

from tal.config import get_resources
from tqdm import tqdm
import concurrent.futures
import multiprocessing as mp

def backproject(H_0, laser_grid_xyz, sensor_grid_xyz, volume_xyz,
				camera_system, t_accounts_first_and_last_bounces,
				t_start, delta_t,
				laser_xyz=None, sensor_xyz=None, progress=False):
	# TODO(oscar): extend for multiple laser points (will not be done)
	# TODO(oscar): Using torchscript but it works as fast as without it just due to GPU and tensorization
	assert H_0.ndim == 2 and laser_grid_xyz.size == 3, \
		'backproject only supports one laser point'
	assert not camera_system.is_transient(), "Transient camera system is not implemented"

	nt, ns = H_0.shape
	assert sensor_grid_xyz.shape[0] == ns, 'H does not match with sensor_grid_xyz'
	assert (not t_accounts_first_and_last_bounces or (laser_xyz is not None and sensor_xyz is not None)), \
		't_accounts_first_and_last_bounces requires laser_xyz and sensor_xyz'
	ns, _ = sensor_grid_xyz.shape
	nv, _ = volume_xyz.shape

	# d_1: laser origin to laser illuminated point
	# d_2: laser illuminated point to x_v
	# d_3: x_v to sensor imaged point
	# d_4: sensor imaged point to sensor origin
	x_l = laser_grid_xyz.reshape(3)
	if t_accounts_first_and_last_bounces:
		d_1 = np.linalg.norm(laser_xyz - x_l)
		d_4 = np.linalg.norm(sensor_grid_xyz - sensor_xyz[np.newaxis,np.newaxis,:])
	else:
		d_1 = np.array([0])
		d_4 = np.zeros(sensor_grid_xyz.shape[0])

	d_14s = d_1 + d_4 - t_start

	reconstruction = __backproject_in_torch(H_0, x_l, sensor_grid_xyz, volume_xyz,
						   d_14s, delta_t, nt, nv)
	gc.collect()
	torch.cuda.empty_cache()
	return reconstruction

import torch
def __backproject_in_torch(H_0, x_l, sensor_grid_xyz, volume_xyz,
							d_14s, delta_t, nt, nv):
	H_0_torch = torch.from_numpy(H_0).cuda()
	x_l_torch = torch.from_numpy(x_l).cuda()
	sensor_grid_xyz_torch = torch.from_numpy(sensor_grid_xyz).cuda()
	volume_xyz_torch = torch.from_numpy(volume_xyz).cuda()
	d_14s_torch = torch.from_numpy(d_14s).cuda()
	delta_t_torch = torch.tensor(delta_t).cuda()
	nt_torch = torch.tensor(nt).cuda()
	nv_torch = torch.tensor(nv).cuda()

	reconstruction = __backproject_parallel(H_0_torch, x_l_torch, sensor_grid_xyz_torch,
			volume_xyz_torch, d_14s_torch, delta_t_torch,
			nt_torch, nv_torch)
	return reconstruction.cpu().numpy()

def __backproject_parallel(H_0, x_l, sensor_grid_xyz, volume_xyz,
					d_14s, delta_t, nt, nv):
	H_1 = torch.zeros(nv, dtype=H_0.dtype, device='cuda')
	s_range = torch.arange(0, sensor_grid_xyz.size(dim=0), dtype=torch.long, device='cuda')

	def work(subrange_v):
		for i_v in subrange_v:
			x_v = volume_xyz[i_v]
			x_s = sensor_grid_xyz
			d_2 = torch.norm(x_l - x_v)
			d_3 = torch.norm(x_v[None,:] - x_s, dim=1)
			t_i = ((d_2 + d_3 + d_14s) / delta_t).long()
			mask = torch.logical_and(t_i >= 0, t_i < nt)
			H_1[i_v] = torch.sum(H_0[t_i[mask], s_range[mask]])

	max_cpu_cores = get_resources().cpu_processes
	max_cpu_cores = mp.cpu_count() if max_cpu_cores == 'max' else max_cpu_cores
	tasks_per_cpu_core = nv // max_cpu_cores
	remaining_tasks = nv % max_cpu_cores
	with concurrent.futures.ThreadPoolExecutor(max_workers=max_cpu_cores) as executor:
		tasks = {executor.submit(work, (tasks_per_cpu_core+1)) : f'i' for i in range(max_cpu_cores)}
		for i in range(max_cpu_cores):
			ntasks = tasks_per_cpu_core if i >= remaining_tasks else (tasks_per_cpu_core+1)
			start = i * tasks_per_cpu_core + min(remaining_tasks, i)
			subrange_v = range(start, start + ntasks)
			tasks[executor.submit(work, subrange_v)] = f'{i}'
		with tqdm(total=max_cpu_cores) as pbar:
			for future in concurrent.futures.as_completed(tasks):
				name = tasks[future]
				pbar.update()
	return H_1
