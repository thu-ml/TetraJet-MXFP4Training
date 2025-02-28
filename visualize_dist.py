import os
from multiprocessing import Pool, Process, spawn
import multiprocessing
from quantization.utils import visualize_distribution

print("input result name(e.g.90epoch):", end=' ')
name = input()
pic_path = os.path.join("visualize", f"pt_{name}")

AWdirs = sorted(os.listdir(pic_path))
for AWdir in AWdirs:
    AWdir = os.path.join(pic_path, AWdir)
    if os.path.isfile(AWdir):
        continue
    FBdirs = sorted(os.listdir(AWdir))
    for FBdir in FBdirs:
        # processes = {}
        FBdir = os.path.join(AWdir, FBdir)
        if os.path.isfile(FBdir):
            continue
        
        pts = sorted(os.listdir(FBdir))

        # with Pool(processes=10) as pool:
        if True:
            results = []
            processes = []
            for pt_path in pts:
                if pt_path.endswith("pt"):
                    processes.append(Process(target=visualize_distribution, args=(os.path.join(FBdir, pt_path), name)))
                    # visualize_distribution(os.path.join(FBdir, pt_path))
                    # result = pool.apply_async(visualize_distribution, (os.path.join(FBdir, pt_path),))
                    # results.append(result)
            num_process = 30
            for i in range(0, len(processes), num_process):
                print(len(processes), f" is processes length at {FBdir}")
                for p in processes[i: i + num_process]:
                    p.start()
                for p in processes[i: i + num_process]:
                    p.join()

            # for result in results:
            #     result.get()
            # pool.join()
            # import IPython
            # IPython.embed()
            # for p in processes.values():
            #     p.start()
            # for idx, (k, p) in enumerate(processes.items()):
            #     p.join()
            #     print(" " * 100, f"The {idx} process named {k} joined")
            #
            # print("@" * 1000, FBdir)

# import IPython

# IPython.embed()
