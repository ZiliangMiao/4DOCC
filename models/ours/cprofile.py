import cProfile
import ray_intersection_cuda

if __name__ == '__main__':
    cProfile.run('ray_intersection_cuda.main()')