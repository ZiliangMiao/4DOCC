import cProfile
import preprocess_nusc_top_script

if __name__ == '__main__':
    cProfile.run('ray_intersection_all_cuda.main()')