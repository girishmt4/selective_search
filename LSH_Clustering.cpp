
#include "falconn/lsh_nn_table.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <boost/tokenizer.hpp>
#include <cstdio>
#include <list>
#include <ctime>
#include <unordered_set>
#include <typeinfo>
#include <math.h>

#include "FALCONN/falconn/src/include/falconn/falconn_global.h"


using namespace std;
using std::cerr;
using std::cout;
using std::endl;
using std::exception;
using std::make_pair;
using std::max;
using std::mt19937_64;
using std::pair;
using std::runtime_error;
using std::string;
using std::uniform_int_distribution;
using std::unique_ptr;
using std::vector;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

using falconn::construct_table;
using falconn::compute_number_of_hash_functions;
using falconn::DenseVector;
using falconn::DistanceFunction;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::LSHNearestNeighborTable;
using falconn::LSHNearestNeighborQuery;
using falconn::QueryStatistics;
using falconn::StorageHashTable;
using falconn::get_default_parameters;


typedef falconn::SparseVector<float> Point;

// const int NUM_QUERIES = 50;
// const int NUM_HASH_TABLES = 10;
// const int NUM_HASH_BITS = 18;
const int NUM_ROTATIONS = 2;
const int FEAT_HASH_DIMENSION = 2048;
const int NUM_SETUP_THREADS = 0;
const int MAX_NEIGHBORS_TO_CHECK = 50;
const float MAX_OVERLAP_TO_CHECK = 0.7;


//Reading the entire dataset from txt file
void read_dataset(std::string file_name, std::vector<Point> *dataset, int* vocabDimension, unordered_map<int,int> *non_empty_mapping, std::list<int> *empty_documents)
{
  std::ifstream infile(file_name);
  std::string line;
  std::getline(infile, line);
  *vocabDimension=stoi(line);

  int index,counter=0;
  int non_empty_counter=0;
  float value;
  typedef boost::tokenizer<boost::char_separator<char> > tokenizer;

  //defining char_separator classes
  boost::char_separator<char> sep_comma(",");
  boost::char_separator<char> sep_dash(":");
  Point p;
  tokenizer::iterator pairs_iter,tok_iter;

  dataset->clear();
  //reading one point at a time
  while (std::getline(infile, line))
  {
    p.clear();
    if(!line.empty())
    {
      //tokenizing the pairs of (index,tfidf_value)
      tokenizer pairs(line, sep_comma);
      for (pairs_iter = pairs.begin();pairs_iter != pairs.end(); ++pairs_iter)
      {
        //tokenizing index and tfidf value from each other
        if(*pairs_iter != "")
        {
          tokenizer tokens(*pairs_iter,sep_dash);
          tok_iter = tokens.begin();
          index = stoi(*tok_iter);

          tok_iter++;
          value = stof(*tok_iter);
          //appending the index & tfidf_value to the data point
          p.push_back(std::make_pair(index,value));
        }

      }
      non_empty_mapping->insert({non_empty_counter,counter});
      non_empty_counter++;

      //appending the data point to the dataset vector
      dataset->push_back(p);

    }
    else{
      // empty_counter++;
      empty_documents->push_back(counter);
      // std::cout << "empty file " << counter << '\n';
    }
    counter++;
    if(counter%1000==0)
    {
      std::cout << counter << " points read" << '\n';
    }
    // std::cout << "Points Read = " << counter << '\n';
  }
  std::cout << "Points Read = " << counter << '\n';
  std::cout << "------------------------" << '\n';

  // for(std::vector<Point>::iterator dataset_it=dataset->begin();dataset_it!=dataset->end();dataset_it++)
  // {
  //   for(Point::iterator point_it = (*dataset_it).begin();point_it != (*dataset_it).end();point_it++)
  //   {
  //     std::cout << point_it->first << " : " << point_it->second << '\n';
  //   }
  //   std::cout << "------------------------" << '\n';
  // }
}

void gen_test_queries(std::vector<Point> *dataset, std::vector<std::vector<Point>> *test_queries, int NUM_QUERIES)
{
  test_queries->clear();
  for(int j=0;j<10;j++)
  {
    std::vector<Point> this_set;
    for (int i = 0; i < NUM_QUERIES; ++i)
    {
      //generating random index to be taken out from the dataset as a query
      srand((unsigned)time(0));
      int ind = rand() % (dataset->size()-1);

      //apending the query to the queries vector
      this_set.push_back((*dataset)[ind]);

      //removing the query point from the dataset
      (*dataset)[ind] = dataset->back();
      dataset->pop_back();
    }
    test_queries->push_back(this_set);
  }

}


//generating queries randomly and separating them from the dataset
void gen_queries(std::vector<Point> *dataset, std::vector<Point> *queries, int NUM_QUERIES)
{
  queries->clear();
  for (int i = 0; i < NUM_QUERIES; ++i)
  {
    //generating random index to be taken out from the dataset as a query
    srand((unsigned)time(0));
    int ind = rand() % (dataset->size()-1);

    //apending the query to the queries vector
    queries->push_back((*dataset)[ind]);

    //removing the query point from the dataset
    (*dataset)[ind] = dataset->back();
    dataset->pop_back();
  }
}



//generates the answers for the queries by linear scan
void gen_answers(const std::vector<Point> &dataset, const std::vector<Point> &queries, std::vector<int> *answers)
{
  answers->clear();
  answers->resize(queries.size());
  int int_index,inner_counter,outer_counter = 0,query_count=0;
  float best,score;
  std::list<int> query_indices;
  std::list<int>::iterator index_found;
  Point::const_iterator query_iterator,datapoint_iterator;

  for (const auto &query : queries)
  {
    best = 0;
    inner_counter = 0;
    query_count++;
    query_indices.clear();

    //storing the indices of the non-zero values of this query
    for(query_iterator=query.begin();query_iterator!=query.end();query_iterator++)
    {
      query_indices.push_back(query_iterator->first);
    }

    for (const auto &datapoint : dataset)
    {
      score=0;

      //calculating the dot products of query and each of the data point
      for(datapoint_iterator=datapoint.begin();datapoint_iterator!=datapoint.end();datapoint_iterator++)
      {
        //checking if the index of non-zero value from data point is present in the query point as well
        index_found=std::find(query_indices.begin(),query_indices.end(),(int)datapoint_iterator->first);

        if(index_found != query_indices.end())
        {
          //calculating integer index of the non zero value
          int_index=std::distance(query_indices.begin(),index_found);

          //incrementing the dot product
          score+=(datapoint_iterator->second)*(query[int_index].second);
        }
      }

      //storing the closest point index to the answers vector i.e. having highest inner product
      if (score > best)
      {
        (*answers)[outer_counter] = inner_counter;
        // std::cout << (*answers)[outer_counter] << '\n';
        best = score;
      }
      ++inner_counter;
    }
    ++outer_counter;
  }
}


//normalizes the data points
void normalize(std::vector<Point> *dataset)
{
  Point::iterator datapoint_iterator;
  for (auto &datapoint : *dataset) {
    float mag = 0;

    //calculating the magnitude of the vector
    for(datapoint_iterator=datapoint.begin();datapoint_iterator!=datapoint.end();datapoint_iterator++)
    {
      mag += (datapoint_iterator->second)*(datapoint_iterator->second);
    }
    mag = sqrt(mag);

    //normalizing the vector
    for(datapoint_iterator=datapoint.begin();datapoint_iterator!=datapoint.end();datapoint_iterator++)
    {
      datapoint_iterator->second = (datapoint_iterator->second)/mag;
    }
  }
}


//evaluate the number of probes so that the precision is greater than 0.9
double evaluate_num_probes(falconn::LSHNearestNeighborTable<Point> *table,const std::vector<Point> &queries,const std::vector<int> &answers, int num_probes)
{
  std::unique_ptr<falconn::LSHNearestNeighborQuery<Point>> query_object = table->construct_query_object(num_probes);
  int outer_counter = 0;
  int num_matches = 0;
  std::vector<int32_t> candidates;

  for (const auto &query : queries)
  {
    //retrieve all the candidate neighbors from this probing sequence along with the duplicates
    query_object->get_candidates_with_duplicates(query, &candidates);
    for (auto x : candidates)
    {
      //checking if the candidates list contain the closest data point calculated by linear scan
      if (x == answers[outer_counter])
      {
        ++num_matches;
        break;
      }
    }
    ++outer_counter;
  }
  return (num_matches + 0.0) / (queries.size() + 0.0);
}


//finding the least number of probes so that success probability is more than 0.9
int find_num_probes(falconn::LSHNearestNeighborTable<Point> *table, const std::vector<Point> &queries, const std::vector<int> &answers, int start_num_probes)
{
  int num_probes = start_num_probes;
  auto t1=std::chrono::high_resolution_clock::now();
  double elapsed_time;
  for (;;)
  {
    std::cout << "\nTrying " << num_probes << " probes...." << '\n';
    //checking if this number of probes gives desired accuracy or not
    double precision = evaluate_num_probes(table, queries, answers, num_probes);
    std::cout << "precision is : " << precision << '\n';
    auto t2 = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

    std::cout << "time taken : " << elapsed_time << '\n';
    if (precision >= 0.9)
    {
      break;
    }

    //if precision is less than 0.9 then doubling the number of probes
    num_probes *= 2;
  }

  int r = num_probes;
  int l = r / 2;

  while (r - l > 1)
  {
    int num_probes = (l + r) / 2;
    std::cout << "\nTrying " << num_probes << " probes...." << '\n';


    //checking if this number of probes gives desired accuracy or not
    double precision = evaluate_num_probes(table, queries, answers, num_probes);
    std::cout << "precision is : " << precision << '\n';
    auto t2 = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

    std::cout << "time taken : " << elapsed_time << '\n';

    //once the precision is 0.9 then finding the least number of probes for 0.9 accuracy
    if (precision >= 0.9)
    {
      r = num_probes;
    }
    else
    {
      l = num_probes;
    }
  }
  return r;
}



void calculateSD_mean(float data[], float *standardDeviation, float *mean)
{
    float sum = 0.0;

    int i;

    for(i = 0; i < 10; ++i)
    {
        sum += data[i];
    }

    *mean = sum/10;

    for(i = 0; i < 10; ++i)
    {
      *standardDeviation += pow(data[i] - (*mean), 2);
    }

    *standardDeviation = *standardDeviation/10;
}

//Usage function for making the user understand the usage of the program
void usage()
{
  std::cout << "usage: ";
  std::cout << "LSH_Sparse dataset_file output_file no_of_queries cluster_limit neighbors_range No_of_Hash_Tables" << '\n';
  std::cout << "Arguments:" << '\n';
  std::cout << "    dataset_file    : The dataset file output by TFIDF_Vectorizer (NOTE : Include the path as well if in other than root directory)" << '\n';
  std::cout << "    output_file     : The output cluster file (NOTE : Include the path as well if in other than root directory)" << '\n';
  std::cout << "    no_of_queries   : No of queries for testing" << '\n';
  std::cout << "    cluster_limit   : Mention the Clusters limit" << '\n';
  std::cout << "    min_neighbors   : Neighbors Range" << '\n';
  std::cout << "    min_overlap     : Overlap Range" << '\n';
  std::cout << "    No_of_Hash_Tables : No. of Hash Tables to build LSH table" << '\n';

}


int main(int argc, char** argv)
{
  if(argc<8)
  {
    usage();
    return -1;
  }

  std::string FILE_NAME=argv[1];
  std::string output_file=argv[2];
  int NUM_QUERIES = stoi(argv[3]);
  size_t cluster_limit = stoi(argv[4]);
  int neighbors_range = stoi(argv[5]);
  float overlap_range = stof(argv[6]);
  const int NUM_HASH_TABLES = stoi(argv[7]);
  int vocabdimension=0,replication_total=0,c_id=0;
  int randomIndex,test_query_counter,flag,absorbant_cluster,overlap,iter,int_index,index_count;
  size_t counter=0;
  float accurate_results,max_overlap,overlap_percentage;
  float accuracy[10];
  float mean=0.0,standardDeviation=0.0;
  double elapsed_time, total_time;

  std::vector<Point> dataset, dataset_temp, dataset_new, dataset_old, queries, sampling_space, test_queries;
  std::vector<Point>::iterator queryIndex,neighborIndex,docs_iterator;
  // Point::iterator datapoint_iterator;
  Point::iterator datapoint_iterator;
  Point average,final_average;

  unordered_map<int,string> ID_point_mapping;
  std::vector<int> answers, test_queries_answers;
  std::vector<int32_t> neighbors,test_query_candidates;
  std::list<int> vocab_indices,empty_documents,query_indices,clustered_docs;
  std::list<int>::iterator empty_documents_iterator,index_found;
  unordered_map<int,int> non_empty_mapping,replication_vector;
  unordered_map<int,int>::iterator replication_vector_iterator;
  unordered_set<int> this_new_cluster;
  map<int,float> neighbors_with_similarity;
  map<int,int> docs_clusters,clusters_sizes;
  map<int,unordered_set<int>> Clusters_new,Clusters_old,Clusters_temp;
  map<int,unordered_set<int>>::iterator Clusters_old_iterator,Clusters_new_iterator;
  map<int,unordered_set<int>>::iterator ClusterIterator;
  map<int,std::pair<int,float>> doc_cluster_similarity;
  map<int,std::pair<int,float>>::iterator doc_cluster_similarity_iterator;
  map<int,int> cluster_dataset_mapping;
  unordered_set<int>::iterator new_cluster_search_key,search_key,cluster_documents;
  unordered_map<int,int>::iterator singleton_clusters_mappings_iterator;
  std::list<int> singleton_clusters;
  unordered_map<int,int> singleton_clusters_mappings,Clusters_mapping;

  ofstream outfile;

  falconn::LSHConstructionParameters params;
  std::unique_ptr<falconn::LSHNearestNeighborQuery<Point>> query_object;

  try
  {
    auto t_start = std::chrono::high_resolution_clock::now();

    // read the dataset
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Reading points from the dataset....\n" << '\n';
    read_dataset(FILE_NAME, &dataset, &vocabdimension, &non_empty_mapping, &empty_documents);
    auto t2 = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    std::cout << "\n" << dataset.size() << " points read in " << elapsed_time << " s" << '\n';
    std::cout << "Non empty = " << non_empty_mapping.size() << '\n';
    std::cout << "Empty = " << empty_documents.size() << '\n';
    for(auto const& i: empty_documents)
    {
      std::cout << i << '\n';
    }
    // std::cout << "Total = " << dataset.size() << '\n';
    std::cout << "---------------------------------------------------------------------" << '\n';

    //normalizing the data points
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "\nNormalizing the points...." << '\n';
    normalize(&dataset);
    t2 = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    std::cout << "Done in " << elapsed_time << " s" << '\n';
    std::cout << "---------------------------------------------------------------------" << '\n';

    //find the center of the mass

    Point sum = dataset[0];
    for (size_t i = 1; i < dataset.size(); ++i)
    {
      for(datapoint_iterator=dataset[i].begin();datapoint_iterator!=dataset[i].end();datapoint_iterator++)
      {
        index_found=std::find(vocab_indices.begin(),vocab_indices.end(),(int)(datapoint_iterator->first));
        if(index_found != vocab_indices.end())
        {
          int_index=std::distance(vocab_indices.begin(),index_found);
          sum[int_index].second += datapoint_iterator->second;
        }
        else
        {
          sum.push_back(std::make_pair(datapoint_iterator->first,datapoint_iterator->second));
          vocab_indices.push_back(datapoint_iterator->first);
        }
      }
    }
    Point center;
    for(datapoint_iterator=sum.begin();datapoint_iterator!=sum.end();datapoint_iterator++)
    {
      center.push_back(std::make_pair(datapoint_iterator->first,datapoint_iterator->second/dataset.size()));
    }

    std::cout << "center calculated" << '\n';

    for (size_t i = 0; i < dataset.size(); ++i)
    {
      for(datapoint_iterator=dataset[i].begin();datapoint_iterator!=dataset[i].end();datapoint_iterator++)
      {
        index_found=std::find(vocab_indices.begin(),vocab_indices.end(),(int)(datapoint_iterator->first));
        if(index_found != vocab_indices.end())
        {
          int_index=std::distance(vocab_indices.begin(),index_found);
          int datapoint_index=std::distance(dataset[i].begin(),datapoint_iterator);
          (dataset[i])[datapoint_index].second -= center[int_index].second;
        }
      }
    }





    //   center += dataset[i];
    // }
    // center /= dataset.size();

    // // re-centering the data to make it more isotropic
    // cout << "re-centering" << endl;
    // for (auto &datapoint : dataset) {
    //   datapoint -= center;
    // }
    // for (auto &query : queries) {
    //   query -= center;
    // }
    // cout << "done" << endl;

    //making a copy of original dataset
    dataset_new = dataset;

    // selecting NUM_QUERIES data points as queries
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "\nSelecting " << NUM_QUERIES << " queries...." << '\n';
    gen_queries(&dataset_new, &queries, NUM_QUERIES);
    t2 = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    std::cout << "Total Queries = " << queries.size() << '\n';
    std::cout << "Done in " << elapsed_time << " s" << '\n';
    std::cout << "---------------------------------------------------------------------" << '\n';
    std::cout << "\n" << dataset_new.size() << " points in the dataset" << '\n';

    // running the linear scan
    std::cout << "\nRunning linear scan (to generate nearest neighbors)...." << '\n';
    t1 = std::chrono::high_resolution_clock::now();
    gen_answers(dataset_new, queries, &answers);
    t2 = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    std::cout << "Done in " << elapsed_time << " s" << '\n';
    std::cout << elapsed_time / queries.size() << " s per query" << '\n';
    std::cout << "---------------------------------------------------------------------" << '\n';

    int NUM_HASH_BITS = (log2(dataset_new.size())/1)+1;


    // setting parameters and constructing the table
    params.dimension = vocabdimension;
    params.lsh_family = falconn::LSHFamily::CrossPolytope;
    params.l = NUM_HASH_TABLES;
    params.distance_function = falconn::DistanceFunction::EuclideanSquared;
    params.feature_hashing_dimension=FEAT_HASH_DIMENSION;
    falconn::compute_number_of_hash_functions<Point>(NUM_HASH_BITS, &params);
    params.num_rotations = NUM_ROTATIONS;
    params.num_setup_threads = NUM_SETUP_THREADS;
    params.storage_hash_table = falconn::StorageHashTable::BitPackedFlatHashTable;
    // params = get_default_parameters<Point>(dataset.size(),
                                   // dataset[0].size(),
                                   // DistanceFunction::EuclideanSquared,
                                   // true);
    std::cout << "\nBuilding the LSH table...." << '\n';
    t1 = std::chrono::high_resolution_clock::now();
    auto table = falconn::construct_table<Point>(dataset_new, params);
    t2 = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    std::cout << "Done in " << elapsed_time << " s" << '\n';
    std::cout << "construction time: " << elapsed_time << " s" << '\n';
    std::cout << "---------------------------------------------------------------------" << '\n';

    // finding the number of probes via the binary search
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "\nFinding the appropriate number of probes...." << '\n';
    // int num_probes = 10;
    int num_probes = find_num_probes(&*table, queries, answers, params.l);
    t2 = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    std::cout << "\nCalculating number of probes done in " << elapsed_time << " s" << '\n';
    // std::cout << "The Optimum number of probes are : " << num_probes << '\n';
    std::cout << "---------------------------------------------------------------------" << '\n';

    //constructing a query object
    query_object = table->construct_query_object(num_probes);

    for(int i=0;i<1;i++)
    {
      std::cout << "---------------------------------------------------Testing LSH Iteration " << (i+1) << '\n';
      // selecting NUM_QUERIES data points as test queries
      t1 = std::chrono::high_resolution_clock::now();
      std::cout << "\nSelecting " << NUM_QUERIES << " test queries...." << '\n';
      gen_queries(&dataset_new, &test_queries, NUM_QUERIES);
      t2 = std::chrono::high_resolution_clock::now();
      elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
      std::cout << "Done in " << elapsed_time << " s" << '\n';

      std::cout << "\n" << dataset_new.size() << " points in the dataset" << '\n';


      // running the linear scan
      std::cout << "\nRunning linear scan (to generate nearest neighbors)...." << '\n';
      t1 = std::chrono::high_resolution_clock::now();
      gen_answers(dataset_new, test_queries, &test_queries_answers);
      t2 = std::chrono::high_resolution_clock::now();
      elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
      std::cout << "Done in " << elapsed_time << " s" << '\n';
      std::cout << elapsed_time / queries.size() << " s per query" << '\n';

      //Evaluating the LSH table
      //Generating the answers for test query set via linear scan
      accurate_results=0;
      test_query_counter=0;
      test_query_candidates.clear();
      for(const auto &test_query : test_queries)
      {
        //retrieve all the candidate neighbors for test query set from this probing sequence along with the duplicates
        query_object->get_candidates_with_duplicates(test_query, &test_query_candidates);
        for(auto x:test_query_candidates)
        {
          //checking if the candidates list contain the closest data point calculated by linear scan
          if (x == test_queries_answers[test_query_counter])
          {
            accurate_results++;
            break;
          }
        }
        test_query_counter++;
      }

      float table_accuracy = accurate_results/NUM_QUERIES;
      std::cout << "Accuracy of the LSH Table is : " << table_accuracy << '\n';
      accuracy[i] = table_accuracy;
    }

    std::cout << "---------------------------------------------------------------------" << '\n';

    calculateSD_mean(accuracy,&standardDeviation,&mean);
    std::cout << "Standard Deviation = " << standardDeviation << '\n';
    std::cout << "Mean = " << mean << '\n';

    //resetting the dataset_new to the original dataset
    dataset_new=dataset;

    std::cout << "---------------------------------------------------------------------" << '\n';
    std::cout << "The Dataset is set back to the original dataset!" << '\n';
    std::cout << "Dataset size = " << dataset_new.size() << '\n';
    std::cout << "---------------------------------------------------------------------" << '\n';

    std::cout << "\nBuilding the LSH table...." << '\n';
    t1 = std::chrono::high_resolution_clock::now();
    table = falconn::construct_table<Point>(dataset_new, params);
    t2 = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    std::cout << "Done in " << elapsed_time << " s" << '\n';
    std::cout << "construction time: " << elapsed_time << " s" << '\n';
    std::cout << "---------------------------------------------------------------------" << '\n';

    // // finding the number of probes via the binary search
    // t1 = std::chrono::high_resolution_clock::now();
    // std::cout << "\nFinding the appropriate number of probes...." << '\n';
    // num_probes = find_num_probes(&*table, queries, answers, params.l);
    // t2 = std::chrono::high_resolution_clock::now();
    // elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    // std::cout << "\nCalculating number of probes done in " << elapsed_time << " s" << '\n';
    // std::cout << "The Optimum number of probes are : " << num_probes << '\n';
    // std::cout << "---------------------------------------------------------------------" << '\n';

    //constructing a query object
    query_object = table->construct_query_object(num_probes);

    auto entire_t1 = std::chrono::high_resolution_clock::now();

    Clusters_new.clear();
    Clusters_old.clear();
    doc_cluster_similarity.clear();

    iter = 1;
    flag = 0;

    //Have a look at this condition
    while(Clusters_new.size() > cluster_limit || iter == 1)
    {
      auto iter_t1 = std::chrono::high_resolution_clock::now();
      std::cout << "\niter " << iter << '\n';

      std::cout << "New Dataset Size = " << dataset_new.size() << '\n';

      //making copies of clusters and datasets
      Clusters_old = Clusters_new;
      dataset_old = dataset_new;

      sampling_space=dataset_new;
      Clusters_new.clear();
      doc_cluster_similarity.clear();
      c_id=0;

      std::cout << "\nRunning the clustering algorithm...." << '\n';
      auto clustering_t1 = std::chrono::high_resolution_clock::now();
      srand((unsigned)time(0));

      while(!sampling_space.empty())
      {
        std::cout << "sampling space size = " << sampling_space.size() << '\n';
        max_overlap=0.0;
        overlap_percentage = 0.0;
        neighbors.clear();
        srand((unsigned)time(0));

        randomIndex = rand() % sampling_space.size();
        int_index=std::distance(dataset_new.begin(),find(dataset_new.begin(),dataset_new.end(),sampling_space[randomIndex]));
        //Mapping can be accessed here
        std::cout << "Index in the dataset = " << int_index << '\n';
        // std::cout << "Int Index = " << randomIndex << '\n';

        if(iter == 1)
        {
          neighbors_with_similarity=query_object->find_k_nearest_neighbors(dataset_new[int_index],neighbors_range,&neighbors);
        }
        else
        {
          neighbors_with_similarity=query_object->find_k_nearest_neighbors(dataset_new[int_index],neighbors_range,&neighbors);
        }

        //calculating the average similarity
        // float average_similarity = 0;
        std::cout << "No of Neighbors = " << neighbors_with_similarity.size() << '\n';
        for(map<int,float>::iterator neighbors_iterator = neighbors_with_similarity.begin();neighbors_iterator != neighbors_with_similarity.end();neighbors_iterator++)
        {
          std::cout << neighbors_iterator->first << " -> " << neighbors_iterator->second << "\n";

          //erasing the neighbors with distance more than 1 units
          // if(neighbors_iterator-> second < -0.33 && neighbors_iterator->first != int_index)
          // {
          //
          //   std::cout << neighbors_iterator->first << "..erased\n";
          //   neighbors_with_similarity.erase(neighbors_iterator->first);
          //   neighbors.erase(find(neighbors.begin(),neighbors.end(),neighbors_iterator->first));
          //
          // }
          // average_similarity = average_similarity + neighbors_iterator->second;
        }
        // average_similarity = average_similarity/neighbors_with_similarity.size();
        // std::cout << "----------------------------------------------" << '\n';

        ClusterIterator = Clusters_new.begin();

        //Check for the overlap percentage
        while(ClusterIterator != Clusters_new.end())
        {
          overlap=0;
          for(auto doc : neighbors)
          {
              search_key = ClusterIterator->second.find(doc);
              if(search_key != ClusterIterator->second.end())
              {
                overlap += 1;
              }
          }
          overlap_percentage = (float)overlap/(float)neighbors.size();
          if(overlap_percentage > max_overlap)
          {
            absorbant_cluster = ClusterIterator->first;
            max_overlap = overlap_percentage;
          }
          ClusterIterator++;
        }

        //have a look
        // if(average_similarity < -0.9)
        // {
        //   max_overlap = 0;
        // }

        if(max_overlap >= overlap_range)
        {

          // std::cout << "merging with cluster " << absorbant_cluster << " & max_overlap = " << max_overlap << '\n';
          for(auto doc : neighbors)
          {
            if(neighbors_with_similarity[doc] > 2)
            {
              continue;
            }
            if(doc_cluster_similarity.find(doc) != doc_cluster_similarity.end())
            {
              if(neighbors_with_similarity[doc] > doc_cluster_similarity[doc].second)
              {
                // std::cout << "similarity of " <<  << '\n';
                Clusters_new[doc_cluster_similarity[doc].first].erase(doc);
                if(Clusters_new[absorbant_cluster].find(doc) == Clusters_new[absorbant_cluster].end())
                {
                  Clusters_new[absorbant_cluster].insert(doc);
                  neighborIndex = find(sampling_space.begin(),sampling_space.end(),dataset_new[doc]);
                  if(neighborIndex!=sampling_space.end())
                  {
                    sampling_space.erase(neighborIndex);
                  }
                  std::cout << doc <<"inserted into cluster " << absorbant_cluster << '\n';
                }
                std::cout << doc << "----" << absorbant_cluster << '\n';
                doc_cluster_similarity[doc].second = neighbors_with_similarity[doc];
                doc_cluster_similarity[doc].first = absorbant_cluster;
              }
            }
            else
            {
              if(Clusters_new[absorbant_cluster].find(doc) == Clusters_new[absorbant_cluster].end())
              {
                Clusters_new[absorbant_cluster].insert(doc);
                neighborIndex = find(sampling_space.begin(),sampling_space.end(),dataset_new[doc]);
                if(neighborIndex!=sampling_space.end())
                {
                  sampling_space.erase(neighborIndex);
                }
                std::cout << doc <<"inserted into cluster " << absorbant_cluster << '\n';
              }
              std::cout << doc << "----" << absorbant_cluster << '\n';
              doc_cluster_similarity.insert(std::make_pair(doc,std::make_pair(absorbant_cluster,neighbors_with_similarity[doc])));
            }

          }
        }
        else
        {

          // std::cout << "creating new cluster " << "max_overlap = " << max_overlap << '\n';
          unordered_set<int> this_cluster;
          int new_cluster_flag=0;
          for(auto doc : neighbors)
          {
            if(neighbors_with_similarity[doc] > 2)
            {
              continue;
            }
            if(doc_cluster_similarity.find(doc) != doc_cluster_similarity.end())
            {
              if(neighbors_with_similarity[doc] > doc_cluster_similarity[doc].second)
              {
                Clusters_new[doc_cluster_similarity[doc].first].erase(doc);
                this_cluster.insert(doc);
                neighborIndex = find(sampling_space.begin(),sampling_space.end(),dataset_new[doc]);
                if(neighborIndex!=sampling_space.end())
                {
                  sampling_space.erase(neighborIndex);
                }
                new_cluster_flag=1;
                std::cout << doc << "----" << c_id << '\n';

                doc_cluster_similarity[doc].second = neighbors_with_similarity[doc];
                doc_cluster_similarity[doc].first = c_id;
              }
            }
            else
            {
              this_cluster.insert(doc);
              neighborIndex = find(sampling_space.begin(),sampling_space.end(),dataset_new[doc]);
              if(neighborIndex!=sampling_space.end())
              {
                sampling_space.erase(neighborIndex);
              }
              new_cluster_flag=1;
              std::cout << doc << "----" << c_id << '\n';
              doc_cluster_similarity.insert(std::make_pair(doc,std::make_pair(c_id,neighbors_with_similarity[doc])));
            }
            // neighborIndex = find(sampling_space.begin(),sampling_space.end(),dataset_new[doc]);
            // if(neighborIndex!=sampling_space.end())
            // {
            //   sampling_space.erase(neighborIndex);
            // }
          }
          if(new_cluster_flag == 1)
          {
            Clusters_new.insert(std::make_pair(c_id,this_cluster));
            c_id += 1;
          }
        }
      }
      // std::cout << "-----------------------before removing duplicates " << iter << "-----------------------" << '\n';
      // for(Clusters_new_iterator=Clusters_new.begin();Clusters_new_iterator!=Clusters_new.end();Clusters_new_iterator++)
      // {
      //   std::cout << "Cluster No. " << Clusters_new_iterator->first << "-------------------------" << '\n';
      //   for(cluster_documents=Clusters_new_iterator->second.begin();cluster_documents!=Clusters_new_iterator->second.end();cluster_documents++)
      //   {
      //     std::cout << (*cluster_documents) << '\n';
      //   }
      // }
      Clusters_new.clear();
      // doc_cluster_similarity.clear();
      // std::cout << "doc_cluster_similarity size = " << doc_cluster_similarity.size() << '\n';
      if(iter == 1)
      {
        for(doc_cluster_similarity_iterator=doc_cluster_similarity.begin();doc_cluster_similarity_iterator!=doc_cluster_similarity.end();doc_cluster_similarity_iterator++)
        {
          // std::cout << doc_cluster_similarity_iterator->first << "-->" << doc_cluster_similarity_iterator->second.first << '\n';
          if(Clusters_new.find(doc_cluster_similarity_iterator->second.first) != Clusters_new.end())
          {
            Clusters_new[(doc_cluster_similarity_iterator->second).first].insert(doc_cluster_similarity_iterator->first);
          }
          else
          {
            unordered_set<int> this_cluster;
            // this_cluster.clear();
            this_cluster.insert(doc_cluster_similarity_iterator->first);
            Clusters_new.insert(make_pair((doc_cluster_similarity_iterator->second).first,this_cluster));
          }
        }
        // for(Clusters_new_iterator=Clusters_new.begin();Clusters_new_iterator!=Clusters_new.end();Clusters_new_iterator++)
        // {
        //   std::cout << "Cluster No. " << Clusters_new_iterator->first << "-------------------------" << '\n';
        //   for(cluster_documents=Clusters_new_iterator->second.begin();cluster_documents!=Clusters_new_iterator->second.end();cluster_documents++)
        //   {
        //     std::cout << (*cluster_documents) << '\n';
        //   }
        // }
      }
      else
      {
        for(doc_cluster_similarity_iterator=doc_cluster_similarity.begin();doc_cluster_similarity_iterator!=doc_cluster_similarity.end();doc_cluster_similarity_iterator++)
        {
          std::cout << doc_cluster_similarity_iterator->first << "-->" << doc_cluster_similarity_iterator->second.first << '\n';
          if(Clusters_new.find(doc_cluster_similarity_iterator->second.first) != Clusters_new.end())
          {
            for(cluster_documents = Clusters_old[doc_cluster_similarity_iterator->first].begin();cluster_documents != Clusters_old[doc_cluster_similarity_iterator->first].end();cluster_documents++)
            {
              Clusters_new[(doc_cluster_similarity_iterator->second).first].insert(*cluster_documents);
              std::cout << *cluster_documents << " inserted into " << doc_cluster_similarity_iterator->second.first << '\n';
            }
          }
          else
          {
            unordered_set<int> this_cluster;
            for(cluster_documents = Clusters_old[doc_cluster_similarity_iterator->first].begin();cluster_documents != Clusters_old[doc_cluster_similarity_iterator->first].end();cluster_documents++)
            {
              this_cluster.insert(*cluster_documents);
              std::cout << *cluster_documents << " inserted into " << doc_cluster_similarity_iterator->second.first << '\n';

            }
            Clusters_new.insert(std::make_pair(doc_cluster_similarity_iterator->second.first,this_cluster));
          }
        }
      }

      std::cout << "-----------------------iter " << iter << "-----------------------" << '\n';
      for(Clusters_new_iterator=Clusters_new.begin();Clusters_new_iterator!=Clusters_new.end();Clusters_new_iterator++)
      {
        std::cout << "Cluster No. " << Clusters_new_iterator->first << "-------------------------" << '\n';
        for(cluster_documents=Clusters_new_iterator->second.begin();cluster_documents!=Clusters_new_iterator->second.end();cluster_documents++)
        {
          std::cout << (*cluster_documents) << '\n';
        }
      }

      auto clustering_t2 = std::chrono::high_resolution_clock::now();
      elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(clustering_t2 - clustering_t1).count();
      std::cout << "clustering algorithm took " << elapsed_time << '\n';
      std::cout << "Total Clusters = " << Clusters_new.size() << '\n';

      // for(std::vector<Point>::iterator dataset_it=dataset_new.begin();dataset_it!=dataset_new.end();dataset_it++)
      // {
      //   for(Point::iterator point_it = (*dataset_it).begin();point_it != (*dataset_it).end();point_it++)
      //   {
      //     std::cout << point_it->first << " : " << point_it->second << '\n';
      //   }
      //   std::cout << "------------------------" << '\n';
      // }


      //Average Calculation
      Clusters_old = Clusters_new;
      dataset_old = dataset_new;

      dataset_new.clear();
      int clusters_count=0;
      Clusters_old_iterator = Clusters_old.begin();
      std::cout << "old dataset size : " << dataset_old.size() << '\n';
      std::cout << "old clusters : " << Clusters_old.size() << '\n';
      //Calculate average and save them in dataset_new
      std::cout << "\nCreating New Dataset (By Taking Averages of Clusters)....\n" << '\n';

      t1 = std::chrono::high_resolution_clock::now();

      for(Clusters_old_iterator=Clusters_old.begin();Clusters_old_iterator!=Clusters_old.end();Clusters_old_iterator++)
      {
        if(Clusters_old_iterator->second.size() == 1)
        {
          singleton_clusters.push_back(Clusters_old_iterator->first);
        }
        final_average.clear();
        average.clear();
        query_indices.clear();
        for(cluster_documents=Clusters_old_iterator->second.begin();cluster_documents!=Clusters_old_iterator->second.end();cluster_documents++)
        {
          // std::cout << "document no. " << *docs_iterator << '\n';
          for(datapoint_iterator=dataset[(*cluster_documents)].begin();datapoint_iterator!=dataset[(*cluster_documents)].end();datapoint_iterator++)
          {
            // std::cout << datapoint_iterator->first << "--" << datapoint_iterator->second << '\n';
            index_found=std::find(query_indices.begin(),query_indices.end(),(int)(datapoint_iterator->first));
            if(index_found != query_indices.end())
            {
              int_index=std::distance(query_indices.begin(),index_found);
              average[int_index].second += datapoint_iterator->second;
            }
            else
            {
              average.push_back(std::make_pair(datapoint_iterator->first,datapoint_iterator->second));
              query_indices.push_back(datapoint_iterator->first);
            }
          }
        }
        for(datapoint_iterator=average.begin();datapoint_iterator!=average.end();datapoint_iterator++)
        {
          final_average.push_back(std::make_pair(datapoint_iterator->first,datapoint_iterator->second/Clusters_old_iterator->second.size()));
        }
        // dataset_new[stoi(Clusters_old_iterator->first)] = final_average;
        dataset_new.push_back(final_average);
      }

      // for(std::vector<Point>::iterator dataset_it=dataset_new.begin();dataset_it!=dataset_new.end();dataset_it++)
      // {
      //   for(Point::iterator point_it = (*dataset_it).begin();point_it != (*dataset_it).end();point_it++)
      //   {
      //     std::cout << point_it->first << " : " << point_it->second << '\n';
      //   }
      //   std::cout << "------------------------" << '\n';
      // }

      t2 = std::chrono::high_resolution_clock::now();
      std::cout << "dataset_new size : " << dataset_new.size() << '\n';
      elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
      std::cout << "New dataset is created in " << elapsed_time << '\n';

      // for(Clusters_new_iterator=Clusters_new.begin();Clusters_new_iterator!=Clusters_new.end();Clusters_new_iterator++)
      // {
      //   std::cout << "Cluster No. " << Clusters_new_iterator->first << "-------------------------" << '\n';
      //   for(cluster_documents=Clusters_new_iterator->second.begin();cluster_documents!=Clusters_new_iterator->second.end();cluster_documents++)
      //   {
      //     std::cout << (*cluster_documents) << '\n';
      //   }
      // }

      if(iter == 1)
      {
        dataset_temp = dataset_new;

        // selecting NUM_QUERIES data points as queries
        t1 = std::chrono::high_resolution_clock::now();
        std::cout << "\nSelecting " << NUM_QUERIES << " queries...." << '\n';
        gen_queries(&dataset_temp, &queries, NUM_QUERIES);
        t2 = std::chrono::high_resolution_clock::now();
        elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        std::cout << "Total Queries = " << queries.size() << '\n';
        std::cout << "Done in " << elapsed_time << " s" << '\n';
        std::cout << "---------------------------------------------------------------------" << '\n';

        std::cout << "\nRunning linear scan (to generate nearest neighbors)...." << '\n';
        t1 = std::chrono::high_resolution_clock::now();
        gen_answers(dataset_temp, queries, &answers);
        t2 = std::chrono::high_resolution_clock::now();
        elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        std::cout << "answers generated in " << elapsed_time << '\n';

        std::cout << "." << '\n';

        std::cout << " Initializing Parameters.... " << '\n';
        t1 = std::chrono::high_resolution_clock::now();
        falconn::compute_number_of_hash_functions<Point>(NUM_HASH_BITS, &params);
        t2 = std::chrono::high_resolution_clock::now();
        elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        std::cout << "parameters initialized in " << elapsed_time << '\n';

        std::cout << "." << '\n';

        std::cout << "Constructing LSH table...." << '\n';
        t1 = std::chrono::high_resolution_clock::now();
        table = falconn::construct_table<Point>(dataset_new, params);
        t2 = std::chrono::high_resolution_clock::now();
        elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        std::cout << "table constructed in " << elapsed_time << '\n';

        std::cout << "." << '\n';

        std::cout << "calculating no. of probes...." << '\n';
        num_probes = find_num_probes(&*table, queries, answers, params.l);
        std::cout << "no of probes calculated" << '\n';

        dataset_temp = dataset_new;
        // Clusters_temp = Clusters_new;

        std::cout << "Constructing LSH table...." << '\n';
        t1 = std::chrono::high_resolution_clock::now();
        table = falconn::construct_table<Point>(dataset_new, params);
        t2 = std::chrono::high_resolution_clock::now();
        elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        std::cout << "table constructed in " << elapsed_time << '\n';

        query_object = table->construct_query_object(num_probes);
        // Clusters_new_iterator = Clusters_new.begin();

        Clusters_old = Clusters_new;
        for(std::list<int>::iterator singleton_iterator=singleton_clusters.begin(); singleton_iterator != singleton_clusters.end();singleton_iterator++)
        {
          Clusters_mapping.insert(make_pair(*singleton_iterator,*singleton_iterator));
        }

        int buffer=0,count=-1;
        for(Clusters_new_iterator=Clusters_old.begin();Clusters_new_iterator!=Clusters_old.end();Clusters_new_iterator++)
        {
          // dataset_temp = dataset_new;
          // Clusters_mapping.insert(make_pair(Clusters_new_iterator->first,Clusters_new_iterator->first));
          count+=1;
          if(std::find(singleton_clusters.begin(),singleton_clusters.end(),Clusters_mapping[Clusters_new_iterator->first]) != singleton_clusters.end())
          {
            std::cout << "Constructing LSH table...." << '\n';
            t1 = std::chrono::high_resolution_clock::now();
            table = falconn::construct_table<Point>(dataset_new, params);
            t2 = std::chrono::high_resolution_clock::now();
            elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
            std::cout << "table constructed in " << elapsed_time << '\n';

            query_object = table->construct_query_object(num_probes);


            neighbors_with_similarity = query_object->find_k_nearest_neighbors(dataset_temp[Clusters_mapping[Clusters_new_iterator->first]],2,&neighbors);
            std::cout << "no of neighbors = " << neighbors_with_similarity.size() << '\n';
            for(map<int,float>::iterator neighbors_iterator = neighbors_with_similarity.begin();neighbors_iterator != neighbors_with_similarity.end();neighbors_iterator++)
            {
              std::cout << neighbors_iterator->first << " - " << neighbors_iterator->second << '\n';
              if(neighbors_iterator->first != Clusters_new_iterator->first)
              {
                std::unordered_set<int>::iterator it=Clusters_new[Clusters_new_iterator->first].begin();
                Clusters_new[neighbors_iterator->first].insert(*it);

                //Check afterwards
                query_indices.clear();
                for(datapoint_iterator = dataset_new[neighbors_iterator->first].begin();datapoint_iterator != dataset_new[neighbors_iterator->first].end();datapoint_iterator++)
                {
                  query_indices.push_back(datapoint_iterator->first);
                }
                for(datapoint_iterator = dataset[*it].begin();datapoint_iterator != dataset[*it].end();datapoint_iterator++)
                {
                  index_found=std::find(query_indices.begin(),query_indices.end(),(int)(datapoint_iterator->first));
                  if(index_found != query_indices.end())
                  {
                    int_index=std::distance(query_indices.begin(),index_found);
                    (dataset_new[neighbors_iterator->first])[int_index].second = (((dataset_new[neighbors_iterator->first])[int_index].second)*Clusters_old[neighbors_iterator->first].size() + datapoint_iterator->second)/Clusters_new[neighbors_iterator->first].size();
                  }
                  else
                  {
                    dataset_new[neighbors_iterator->first].push_back(make_pair(datapoint_iterator->first,datapoint_iterator->second/Clusters_new[neighbors_iterator->first].size()));
                    query_indices.push_back(datapoint_iterator->first);
                  }
                }

              }
            }
            Clusters_new.erase(Clusters_new_iterator->first);
            dataset_new[count-buffer] = dataset_new.back();
            dataset_new.pop_back();
            buffer += 1;

            // dataset_new[(*singleton_iterator)]


          }
          else
          {
            unordered_set<int> this_cluster = Clusters_new[Clusters_new_iterator->first];
            Clusters_new.erase(Clusters_new_iterator->first);
            Clusters_new.insert(make_pair(count-buffer,this_cluster));
          }
          Clusters_mapping[(Clusters_new_iterator->first)] = count-buffer;
          //PUT A CONDITION HERE (IF MORE SINGLETON CLUSTERS ARE REMAINING)
          //THEN ONLY CONSTRUCT ANOTHER LSH TABLE


        }
        std::cout << "Constructing LSH table...." << '\n';
        t1 = std::chrono::high_resolution_clock::now();
        table = falconn::construct_table<Point>(dataset_new, params);
        t2 = std::chrono::high_resolution_clock::now();
        elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        std::cout << "table constructed in " << elapsed_time << '\n';

        query_object = table->construct_query_object(num_probes);
        // for(std::list<int>::iterator singleton_iterator=singleton_clusters.begin(); singleton_iterator != singleton_clusters.end();singleton_iterator++)
        // {
        //   Clusters_mapping.insert(make_pair(*singleton_iterator,*singleton_iterator));
        //   // Clusters_new_iterator++;
        //   // std::cout << "cluster no. " << *singleton_iterator << '\n';
        //   neighbors_with_similarity = query_object->find_k_nearest_neighbors(dataset_temp[(*singleton_iterator)],2,&neighbors);
        //   std::cout << "no of neighbors = " << neighbors_with_similarity.size() << '\n';
        //   for(map<int,float>::iterator neighbors_iterator = neighbors_with_similarity.begin();neighbors_iterator != neighbors_with_similarity.end();neighbors_iterator++)
        //   {
        //     std::cout << neighbors_iterator->first << " - " << neighbors_iterator->second << '\n';
        //     if(neighbors_iterator->first != *singleton_iterator)
        //     {
        //       std::unordered_set<int>::iterator it=Clusters_new[*singleton_iterator].begin();
        //       std::cout << "dataset size is = " << dataset_new.size() << '\n';
        //       // std::cout << " is = " << dataset_new.size() << '\n';
        //       Clusters_new[neighbors_iterator->first].insert(*it);
        //
        //       query_indices.clear();
        //       for(datapoint_iterator = dataset_new[neighbors_iterator->first].begin();datapoint_iterator != dataset_new[neighbors_iterator->first].end();datapoint_iterator++)
        //       {
        //         query_indices.push_back(datapoint_iterator->first);
        //       }
        //       for(datapoint_iterator = dataset[*it].begin();datapoint_iterator != dataset[*it].end();datapoint_iterator++)
        //       {
        //         index_found=std::find(query_indices.begin(),query_indices.end(),(int)(datapoint_iterator->first));
        //         if(index_found != query_indices.end())
        //         {
        //           int_index=std::distance(query_indices.begin(),index_found);
        //           (dataset_new[neighbors_iterator->first])[int_index].second = (((dataset_new[neighbors_iterator->first])[int_index].second)*Clusters_old[neighbors_iterator->first].size() + datapoint_iterator->second)/Clusters_new[neighbors_iterator->first].size();
        //         }
        //         else
        //         {
        //           dataset_new[neighbors_iterator->first].push_back(make_pair(datapoint_iterator->first,datapoint_iterator->second/Clusters_new[neighbors_iterator->first].size()));
        //           query_indices.push_back(datapoint_iterator->first);
        //         }
        //       }
        //     }
        //   }
        //
        //   // Clusters_old = Clusters_new;
        //
        //   int buffer=0,count=-1;
        //   for(Clusters_new_iterator=Clusters_new.begin();Clusters_new_iterator!=Clusters_new.end();Clusters_new_iterator++)
        //   {
        //     //change the cluster ID
        //     count += 1;
        //     if(std::find(singleton_clusters.begin(),singleton_clusters.end(),Clusters_new_iterator->first) != singleton_clusters.end())
        //     {
        //       //delete from the clusters
        //       Clusters_new.erase(Clusters_new_iterator->first);
        //       dataset_new[count-buffer] = dataset_new.back();
        //       dataset_new.pop_back();
        //       buffer += 1;
        //       //delete from the dataset_new
        //       // continue;
        //     }
        //     else
        //     {
        //       // unordered_set<int> this_cluster;
        //       Clusters_mapping[(*singleton_iterator)] = count-buffer;
        //       // Clusters_mapping.insert(make_pair(Clusters_new_iterator->first,count-buffer));
        //       Clusters_new.insert(make_pair(count-buffer,Clusters_new_iterator->second));
        //       Clusters_new.erase(Clusters_new_iterator->first);
        //       // Clusters_new.insert()
        //       // Clusters_new_iterator->first = count-buffer;
        //       // singleton_clusters_mappings.insert(make_pair(count,count+buffer));
        //     }
        //   }
        //
        //
        //   //PUT A CONDITION HERE (IF MORE SINGLETON CLUSTERS ARE REMAINING)
        //   //THEN ONLY CONSTRUCT ANOTHER LSH TABLE
        //
        //   // dataset_new[(*singleton_iterator)]
        //   std::cout << "Constructing LSH table...." << '\n';
        //   t1 = std::chrono::high_resolution_clock::now();
        //   table = falconn::construct_table<Point>(dataset_new, params);
        //   t2 = std::chrono::high_resolution_clock::now();
        //   elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        //   std::cout << "table constructed in " << elapsed_time << '\n';
        //
        //   query_object = table->construct_query_object(num_probes);
        // }

        // for(Clusters_new_iterator=Clusters_new.begin();Clusters_new_iterator!=Clusters_new.end();Clusters_new_iterator++)
        // {
        //   if(Clusters_new_iterator->size() == 1)
        //   {
        //
        //   }
        // }
        // for(std::list<int>::iterator singleton_iterator=singleton_clusters.begin(); singleton_iterator != singleton_clusters.end();singleton_iterator++)
        // {
        //   Clusters_new.erase(*singleton_iterator);
        // }


        //Printing Out the clusters again for checking
        std::cout << "new logic done" << '\n';

      }

      for(Clusters_new_iterator=Clusters_new.begin();Clusters_new_iterator!=Clusters_new.end();Clusters_new_iterator++)
      {
        std::cout << "Cluster No. " << Clusters_new_iterator->first << "-------------------------" << '\n';
        for(cluster_documents=Clusters_new_iterator->second.begin();cluster_documents!=Clusters_new_iterator->second.end();cluster_documents++)
        {
          std::cout << (*cluster_documents) << '\n';
        }
      }

      std::cout << "." << '\n';
      auto iter_t2 = std::chrono::high_resolution_clock::now();
      elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(iter_t2 - iter_t1).count();
      std::cout << "iter " << iter << " took " << elapsed_time << '\n';
      iter++;


    }

    // for(Clusters_new_iterator=Clusters_new.begin();Clusters_new_iterator!=Clusters_new.end();Clusters_new_iterator++)
    // {
    //   std::cout << "Cluster No. " << Clusters_old_iterator->first << "-------------------------" << '\n';
    //   for(cluster_documents=Clusters_old_iterator->second.begin();cluster_documents!=Clusters_old_iterator->second.end();cluster_documents++)
    //   {
    //     std::cout << (*cluster_documents) << '\n';
    //   }
    // }

    for(Clusters_new_iterator = Clusters_new.begin();Clusters_new_iterator != Clusters_new.end();Clusters_new_iterator++)
    {
      // std::cout << "Cluster no. " << (Clusters_new_iterator->first) << '\n';
      for(cluster_documents = Clusters_new_iterator->second.begin();cluster_documents!=Clusters_new_iterator->second.end();cluster_documents++)
      {
        docs_clusters.insert(make_pair((*cluster_documents),Clusters_new_iterator->first));
        // outfile <<  << ":" <<  << "\n";
        replication_total += 1;
        // std::cout << "Document no. " << *cluster_documents << '\n';
        counter++;
        replication_vector_iterator = replication_vector.find(*cluster_documents);
        if(replication_vector_iterator != replication_vector.end())
        {
          replication_vector[(*cluster_documents)] += 1;
        }
        else{
          replication_vector.insert(std::make_pair((*cluster_documents),1));
        }
      }
    }

    //To be deleted
    //just to find out the difference between the farthest Points
    // selecting NUM_QUERIES data points as queries
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "\nSelecting " << NUM_QUERIES << " queries...." << '\n';
    gen_queries(&dataset_temp, &queries, NUM_QUERIES);
    t2 = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    std::cout << "Total Queries = " << queries.size() << '\n';
    std::cout << "Done in " << elapsed_time << " s" << '\n';
    std::cout << "---------------------------------------------------------------------" << '\n';

    std::cout << "\nRunning linear scan (to generate nearest neighbors)...." << '\n';
    t1 = std::chrono::high_resolution_clock::now();
    gen_answers(dataset_temp, queries, &answers);
    t2 = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    std::cout << "answers generated in " << elapsed_time << '\n';

    std::cout << "." << '\n';

    std::cout << " Initializing Parameters.... " << '\n';
    t1 = std::chrono::high_resolution_clock::now();
    falconn::compute_number_of_hash_functions<Point>(NUM_HASH_BITS, &params);
    t2 = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    std::cout << "parameters initialized in " << elapsed_time << '\n';

    std::cout << "." << '\n';

    std::cout << "Constructing LSH table...." << '\n';
    t1 = std::chrono::high_resolution_clock::now();
    table = falconn::construct_table<Point>(dataset_new, params);
    t2 = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    std::cout << "table constructed in " << elapsed_time << '\n';

    std::cout << "." << '\n';

    // std::cout << "calculating no. of probes...." << '\n';
    // num_probes = find_num_probes(&*table, queries, answers, params.l);
    // std::cout << "no of probes calculated" << '\n';

    dataset_temp = dataset_new;
    // Clusters_temp = Clusters_new;

    std::cout << "Constructing LSH table...." << '\n';
    t1 = std::chrono::high_resolution_clock::now();
    table = falconn::construct_table<Point>(dataset_new, params);
    t2 = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    std::cout << "table constructed in " << elapsed_time << '\n';

    query_object = table->construct_query_object(100);
    // Clusters_new_iterator = Clusters_new.begin();
    sampling_space=dataset_new;
    randomIndex = rand() % sampling_space.size();
    int_index=std::distance(dataset_new.begin(),find(dataset_new.begin(),dataset_new.end(),sampling_space[randomIndex]));
    std::cout << "Index in the dataset = " << int_index << '\n';


    neighbors_with_similarity=query_object->find_k_nearest_neighbors(sampling_space[randomIndex],5,&neighbors);

    std::cout << "No of Neighbors = " << neighbors_with_similarity.size() << '\n';
    for(map<int,float>::iterator neighbors_iterator = neighbors_with_similarity.begin();neighbors_iterator != neighbors_with_similarity.end();neighbors_iterator++)
    {
      std::cout << neighbors_iterator->first << " -> " << neighbors_iterator->second << "\n";
      // average_similarity = average_similarity + neighbors_iterator->second;
    }


//---------------------------------------------------------------------------
//---------------------------------------------------------------------------


    std::cout << replication_total << '\n';
    std::cout << dataset.size() << '\n';
    std::cout << "Average replication factor is : " << replication_total/dataset.size() << '\n';

    std::cout << counter << '\n';
    int x = counter-(dataset.size());
    std::cout << "\nThe Increase in the no. of documents = " << x << '\n';

    auto entire_t2 = std::chrono::high_resolution_clock::now();

    outfile.open(output_file);

    for(size_t i=0; i<dataset.size();i++)
    {
      outfile << i << ":" << docs_clusters[i] << "\n";
    }

    outfile.close();
    // system(("python3 adjusted_rand_index.py "+output_file).c_str());

  }
  catch (std::runtime_error &e)
  {
    std::cerr << "Runtime error: " << e.what() << '\n';
    return 1;
  }
  catch (std::exception &e)
  {
    std::cerr << "Exception: " << e.what() << '\n';
    return 1;
  }
  catch (...)
  {
    std::cerr << "ERROR" << '\n';
    return 1;
  }
  return 0;
}
