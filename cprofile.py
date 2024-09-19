import cProfile
import preprocess_rays_mutual_obs_script

if __name__ == '__main__':
    cProfile.run('ray_intersection_all_cuda.main()')