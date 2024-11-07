
#include <faiss/AutoTune.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQFastScanDisk.h>

#include <faiss/index_io.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <omp.h>

using idx_t = faiss::idx_t;

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}
void load_data(const char* filename, unsigned char*& data, int num, int dim) { 
  std::ifstream in(filename, std::ios::binary);	//open file in binary
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  data = new unsigned char[(size_t)num * (size_t)dim];
  in.seekg(0, std::ios::beg);	//shift to start

  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);	//right shift 4 bytes
  	for(int j=0;j<dim;j++)
    		in.read((char*)(data + i*dim+j), 1 );	//load data	
   // in.read((char*)(data + i*dim), dim );
  }  
  in.close();
}
void disk_build(const char* base_filepath, int nb, int d, int ratio, const std::string& index_store_path, const std::string& disk_store_path) {
    double t0 = elapsed();
    size_t dd; // dimension
    size_t nt; // number of vectors

    float* xb = fvecs_read(base_filepath, &dd, &nt);
    //     unsigned char* xb_old;
    // xb_old = (unsigned char*)malloc(d * nb ); 
    // load_data(base_filepath, xb_old, nb, d);
    //      float* xb = new float[d * nb];
	//    {    
	// 	   for (int i = 0; i < d*nb; i++) {
    //         xb[i] = xb_old[i];
    //     }
    //     delete[] xb_old;
    // } 

        srand((int)time(0)); 
        std::vector<float> trainvecs(nb/ratio * d);
        for (int i = 0; i < nb/ratio; i++) {
        int rng=(rand() % (nb+1));; //rng=random number in nb

        for (int j = 0; j < d; j++){
            //printf(" setting %d vector's %d data, trianvecs[%d]=xb[%d] \n",i,j,d * i + j,rng * d + j);
            trainvecs[d * i + j] = xb[rng * d + j];
        }
    }

    int nlist = 5000;
    int m = 64;
    int nbits = 4;
    int top_clusters = 20;
    float estimate_factor = 1.2;
    float prune_factor = 3;
    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFPQFastScanDisk index(&quantizer, d, nlist, m, nbits, top_clusters, estimate_factor, prune_factor, index_store_path);
    index.set_assign_replicas(2);
    index.verbose = false;
    omp_set_num_threads(16);
    printf("[%.3f s] IndexIVFPQfs_disk start to train\n", elapsed() - t0);
    index.train(nb / ratio, trainvecs.data());
    printf("[%.3f s] IndexIVFPQfs_disk train finished\n", elapsed() - t0);

    printf("[%.3f s] IndexIVFPQfs_disk start to add\n", elapsed() - t0);
    index.add(nb, xb);
    printf("[%.3f s] IndexIVFPQfs_disk add finished\n", elapsed() - t0);

    printf("[%.3f s] IndexIVFPQfs_disk start to reorg\n", elapsed() - t0);
    index.initial_location(xb);
    printf("[%.3f s] IndexIVFPQfs_disk reorg finished\n", elapsed() - t0);

    // Write index to disk
    faiss::write_index(&index, disk_store_path.c_str());
    printf("[%.3f s] IndexIVFPQfs_disk written to disk\n", elapsed() - t0);

    delete[] xb;
}

void search(const char* query_filepath, const char* ground_truth_filepath, int nq, int d, int k, const std::string& disk_store_path) {
    double t0 = elapsed();
    size_t dd2; // dimension
    size_t nt2; // number of queries
    float* xq = fvecs_read(query_filepath, &dd2, &nt2);
    //     unsigned char* xq_old;
	//    xq_old = (unsigned char*)malloc(d * nq );
    //    load_data(query_filepath, xq_old, nq, d);
    //         float* xq = new float[d * nq];
    //        {       
    //                for (int i = 0; i < d*nq; i++) {
    //         xq[i] = xq_old[i];
    //     }
    //     delete[] xq_old;
    // }

    size_t nq2;
    size_t kk;
    int* gt_int = ivecs_read(ground_truth_filepath, &kk, &nq2);
    float estimate_factor = 1.5;
    float estimate_factor_partial = 1.1;
    float prune_factor = 3;

    faiss::IndexIVFPQFastScanDisk* index = dynamic_cast<faiss::IndexIVFPQFastScanDisk*>(faiss::read_index(disk_store_path.c_str()));
    index->set_top(20);
    index->set_assign_replicas(2);
    index->set_estimate_factor(estimate_factor);
    index->set_estimate_factor_partial(estimate_factor_partial);
    index->set_prune_factor(prune_factor);

    std::vector<double> search_times;
    std::vector<double> recalls;
    omp_set_num_threads(16);
    idx_t* I = new idx_t[k * nq];
    float* D = new float[k * nq];
        //         faiss::idx_t* gt;
        //                 gt = new faiss::idx_t[100 * nq];
        //                 int jj=0;
        // for (int i = 0; i < kk * nq; i++) {
        //     gt[jj] = gt_int[i]; //long int / int
        //     jj+=1;
        //     if((i+1)%100==0){
        //         i+=900;
        //     }

        // }
        // delete[] gt_int;
            faiss::idx_t* gt = new faiss::idx_t[kk * nq];
    for (int i = 0; i < kk * nq; i++) {
        gt[i] = gt_int[i];
    }
    delete[] gt_int;

    //int arr[] = {5, 10};
    //int arr[] = {20,30, 50,60,70,80, 90,100};
    // int arr[] = {20, 30, 40, 60, 80, 100, 120, 140, 160, 180, 240, 300, 340};
    int arr[] = {40, 80, 100, 120, 140, 160, 180, 240, 300, 340};
    //int arr[] = {100};

    for (int i : arr) {
        printf("-----------now, the nprobe is=%d---------\n", i);

        index->nprobe = i;
        // index->search(nq, xq, k, D, I);
        printf("[%.3f s] IndexIVFPQ_disk start to search\n", elapsed() - t0);
        double t1 = elapsed();
        index->search(nq, xq, k, D, I);
        double t2 = elapsed();
        double search_time = t2 - t1;
        printf("[%.3f s] IndexIVFPQ_disk search finished\n", elapsed() - t0);

        size_t fcc = faiss::indexIVFPQFastScanDisk_stats.full_cluster_compare;
        size_t fcr = faiss::indexIVFPQFastScanDisk_stats.full_cluster_rerank;
        size_t pcc = faiss::indexIVFPQFastScanDisk_stats.partial_cluster_compare;
        size_t pcr = faiss::indexIVFPQFastScanDisk_stats.partial_cluster_rerank;
        size_t sl  = faiss::indexIVFPQFastScanDisk_stats.pruned;

        double mepq = faiss::indexIVFPQFastScanDisk_stats.memory_pq_elapsed.count() ; 
        double me2 = faiss::indexIVFPQFastScanDisk_stats.memory_2_elapsed.count() ; 
        double dfe = faiss::indexIVFPQFastScanDisk_stats.disk_full_elapsed.count() ; 
        double dpe = faiss::indexIVFPQFastScanDisk_stats.disk_partial_elapsed.count() ; 
        double oe = faiss::indexIVFPQFastScanDisk_stats.others_elapsed.count() ; 
        double ce = faiss::indexIVFPQFastScanDisk_stats.coarse_elapsed.count() ; 
        double re = faiss::indexIVFPQFastScanDisk_stats.rank_elapsed.count() ; 

        std::cout << "full_cluster_compare      :" << fcc/nq << std::endl;
        std::cout << "full_cluster_rerank       :" << fcr/nq << std::endl;
        std::cout << "partial_cluster_compare   :" << pcc/nq << std::endl;
        std::cout << "partial_cluster_rerank    :" << pcr/nq << std::endl;

        std::cout << "AVG rerank ratio(full)    :" << static_cast<double>(fcr) / fcc << std::endl;
        std::cout << "AVG rerank ratio(partial) :" << static_cast<double>(pcr) / pcc << std::endl;

        std::cout << "Scanned lists total       :" << sl << std::endl;
        std::cout << "Scanned lists per query   :" << static_cast<double>(sl) / nq << std::endl;
        std::cout << "Scanned lists account for :" << static_cast<double>(sl) / (nq * i) << std::endl;

        std::cout << "memory_pq_elapsed        :" << mepq/ 1e6 << std::endl;
        std::cout << "memory_2_elapsed        :" << me2/ 1e6 << std::endl;
        std::cout << "disk_full_elapsed       :" <<dfe/ 1e6 << std::endl;
        std::cout << "disk_partial_elapsed    :" << dpe/ 1e6 << std::endl;
        std::cout << "others_elapsed    :" << oe/ 1e6 << std::endl;
        std::cout << "coarse_elapsed    :" << ce/ 1e6 << std::endl;
        std::cout << "rank_elapsed    :" << re/ 1e6 << std::endl
                    << std::endl
                    << std::endl;

        faiss::indexIVFPQFastScanDisk_stats.reset();

        int repeated = 0;
        for (int i = 0; i < nq; i++)
        {
            std::unordered_set<size_t> seen_ids;

            for (int j = 0; j < k; j++)
            {
                size_t temp_id = I[i * k + j];  // 获取第 i 个查询的第 j 个结果的 ID
                
                if (seen_ids.find(temp_id) != seen_ids.end())
                {
                    repeated++;
                    std::cout << "nq:" << i << "  nr:" << j << std::endl; 
                }
                else
                {
                    seen_ids.insert(temp_id);
                }
            }
        }
        std::cout << "repeated = " << repeated << std::endl;

        // evaluate result
        int n2_100 = 0;
        for (int i = 0; i < nq; i++) {
            std::map<float, int> umap;
            for (int j = 0; j < k; j++) {
                umap.insert({gt[i * 100 + j], 0});
            }
            for (int l = 0; l < k; l++) {
                if (umap.find(I[i * k + l]) != umap.end()) {
                    n2_100++;
                }
            }
            umap.clear();
        }
        printf("Intersection R@100 = %.4f\n", n2_100 / float(nq * k));
        double recall = n2_100 / float(nq * k);
        search_times.push_back(search_time / nq * 1000);
        recalls.push_back(recall);
        if(recall > 0.99)
            break;
    }

    delete[] I;
    delete[] D;
    delete[] xq;
    delete[] gt;

    // Print the search times and recalls
    std::cout << "Search times: [";
    for (size_t i = 0; i < search_times.size(); i++) {
        std::cout << search_times[i];
        if (i < search_times.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    std::cout << "QPS: [";
    for (size_t i = 0; i < search_times.size(); i++) {
        std::cout << (1000.0 / search_times[i]);
        if (i < search_times.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    std::cout << "Recalls: [";
    for (size_t i = 0; i < recalls.size(); i++) {
        std::cout << recalls[i];
        if (i < recalls.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    delete index;
}

int main(int argc, char *argv[]) {
    int type = atoi(argv[1]); // Set to 0 for build, 1 for search

    int d = 128;      // dimension
    int nb = 1000000; // database size
    int nq = 10000;   // number of queries
    int ratio = 20;   // ratio for training
    int k = 100;      // number of nearest neighbors to search
    const char* base_filepath =
            "/ssd_root/zhan4404/dataset/dataset/sift1B/bigann_10m_base.bvecs";
    const char* query_filepath =
            "/ssd_root/zhan4404/dataset/dataset/sift1B/bigann_query.bvecs";
    const char* ground_truth_filepath =
            "/ssd_root/zhan4404/dataset/dataset/sift1B/gnd/idx_10M.ivecs";
        // data
    std::string index_store_path = "/home/zhan4404/disk-based/newfaiss_disk/faiss4_disk_vector/build/demos/index_folder/sift10m/sift10M_ivfpqfsdisk";
        // index like pq.. parameters..
    std::string disk_store_path = "/home/zhan4404/disk-based/newfaiss_disk/faiss4_disk_vector/build/demos/index_folder/sift10m/sift10M_ivfpqfsdisk.index";
    //     const char* base_filepath = "/home/zhan4404/costeff/diskann/DiskANN/build/data/sift/sift_base.fvecs";
    // const char* query_filepath = "/home/zhan4404/costeff/diskann/DiskANN/build/data/sift/sift_query.fvecs";
    // const char* ground_truth_filepath =  "/home/zhan4404/costeff/diskann/DiskANN/build/data/sift/sift_groundtruth.ivecs";
    //     // data
    // std::string index_store_path = "/home/zhan4404/disk-based/newfaiss_disk/faiss4_disk_vector/build/demos/index_folder/sift1m/sift1M_ivfpqfsdisk";
    //     // index like pq.. parameters..
    // std::string disk_store_path = "/home/zhan4404/disk-based/newfaiss_disk/faiss4_disk_vector/build/demos/index_folder/sift1m/sift1M_ivfpqfsdisk.index";

    if (type == 0) {
        disk_build(base_filepath, nb, d, ratio, index_store_path, disk_store_path);
    } else if (type == 1) {
        search(query_filepath, ground_truth_filepath, nq, d, k, disk_store_path);
    }

    return 0;
}
