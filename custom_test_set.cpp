#include <string>
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace std;


void usage()
{
  std::cout << "usage: ";
  std::cout << "Arguments:" << '\n';
  std::cout << "    no_files      : No. of Files" << '\n';
  std::cout << "    no_clusters   : No. of Clusters" << '\n';
  std::cout << "    no_features   : No. of Features" << '\n';
  std::cout << "    output_file   : Output File" << '\n';
  std::cout << "    ground_truth  : Ground Truth File" << '\n';

}

int main(int argc, char** argv)
{

  if(argc<6)
  {
    usage();
    return -1;
  }

  int total_files_count=stoi(argv[1]);
  int total_clusters_count=stoi(argv[2]);
  int total_features_count=stoi(argv[3]);
  string output_file = argv[4];
  string ground_truth_file = argv[5];
  int files_counter=1, clusters_counter=1;
  int no_of_features, feature_to_be_set;
  unordered_set<int> this_file_features;
  int files_per_cluster = total_files_count/total_clusters_count;
  int features_per_cluster = total_features_count/total_clusters_count;

  srand((unsigned)time(0));

  ofstream outfile,truthfile;
  truthfile.open(ground_truth_file);
  outfile.open(output_file);
  outfile << total_features_count << "\n";

  while(clusters_counter <= total_clusters_count)
  {
    files_counter=1;
    std::cout << "---------------------------------Cluster no. " << clusters_counter << '\n';
    while(files_counter <= files_per_cluster)
    {
      truthfile << (clusters_counter-1)*files_per_cluster + files_counter - 1 << ":" << clusters_counter-1 << "\n";
      std::cout << "File No.------------------ " << files_counter << '\n';
      do {
        // no_of_features = 70;
        no_of_features = rand() % features_per_cluster;
      } while(no_of_features < 10);
      std::cout << "no of features = " << no_of_features << '\n';
      int features_set = 1;
      while(features_set <= no_of_features)
      {
        feature_to_be_set = (rand() % features_per_cluster)+(features_per_cluster*(clusters_counter-1));
        std::cout << "feature to be set = " << feature_to_be_set << '\n';
        if(this_file_features.find(feature_to_be_set) == this_file_features.end())
        {

          outfile << feature_to_be_set << ":" << "1" <<",";
          this_file_features.insert(feature_to_be_set);
          features_set++;
        }
      }
      outfile << "\n";
      files_counter++;
      this_file_features.clear();
    }
    clusters_counter++;
  }

  if(total_files_count%total_clusters_count != 0)
  {
    clusters_counter--;
    int remain = total_files_count%total_clusters_count;
    while(remain > 0)
    {
      std::cout << "File No.------------------ " << files_counter << '\n';
      do {
        no_of_features = rand() % features_per_cluster;
      } while(no_of_features == 0);
      std::cout << "no of features = " << no_of_features << '\n';
      int features_set = 1;
      while(features_set <= no_of_features)
      {
        feature_to_be_set = (rand() % features_per_cluster)+(features_per_cluster*(clusters_counter-1));
        std::cout << "feature to be set = " << feature_to_be_set << '\n';
        if(this_file_features.find(feature_to_be_set) == this_file_features.end())
        {
          outfile << feature_to_be_set << "-" << "1,";
          this_file_features.insert(feature_to_be_set);
        }
        features_set++;
      }
      outfile << "\n";
      files_counter++;
      this_file_features.clear();
      remain--;
    }
  }

}
