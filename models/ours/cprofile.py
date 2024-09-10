import cProfile
import ray_intersection
import ray_intersection_all_cuda

if __name__ == '__main__':
    cProfile.run('ray_intersection_all_cuda.main()')