#include <iostream>
#include <fstream>
#include <algorithm>
#include <cctype>
#include<math.h>
#include <vector>
#include <string.h>
#include <ctime>
#include<unordered_set>
#include<map>
#include "porter2_stemmer.h"
#include "porter2_stemmer.cpp"
#include <experimental/filesystem>
#include <boost/tokenizer.hpp>
#include <string>
using namespace std;
namespace fs = std::experimental::filesystem;

//Global Declaration of Stopwords List
unordered_set<string> stop_list;

struct non_alpha {
    bool operator()(char c) {
        return !std::isalpha(c);
    }
};

int vocabularyIndex=0;

//Function checks if the word is a stopword or not
bool is_in_stopwords(string word)
{
  if(stop_list.find(word)!=stop_list.end())
  {
    return true;
  }
  return false;
}

//Function checks if there are any digits in the term
bool has_any_digits(const std::string& s)
{
  return std::any_of(s.begin(), s.end(), ::isdigit);
}

bool has_any_letter(const std::string& s)
{
  return std::any_of(s.begin(),s.end(), non_alpha());
}

//Function reads the stopwords from the stopwords file and stores in a unordered_set
void read_stopwords(string stopfile)
{
  //Reading Stopwords list file
  std::ifstream stoplist(stopfile);
  char stopword[255];
  while(stoplist)
  {
    //Reading the stopwords line by line
    stoplist.getline(stopword,255);
    stop_list.insert(stopword);
  }
  stoplist.close();
}

//Calculates the TF Scores document by document
map<int,float> getTFScores(string file_content_in_string, map<int,float> *DFVector, map<string,int> *Vocabulary)
{
  string word;
  map<int,float> TFScores;
  typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
  boost::char_separator<char> sep(",-\\.<>/?;:'\"{}[] \n\t|`~!@#$%^&*()_=+");
  tokenizer tokens(file_content_in_string, sep);

  for (tokenizer::iterator tok_iter = tokens.begin();tok_iter != tokens.end(); ++tok_iter)
  {
    //filtering numeric or alphanumeric terms
    // if(!has_any_digits(*tok_iter))
    // {
      //checking if the string contains letter
      if(!has_any_letter(*tok_iter))
      {
        //filtering the stopwords
        if(is_in_stopwords(*tok_iter)==false)
        {
          //filtering the terms having length '1'(characters)
          if((*tok_iter).length()>1)
          {
            word=*tok_iter;

            //stemming the term
            Porter2Stemmer::stem(word);

            //if the term is not in DFVector(Unique terms vector i.e. vocabulary)
            //then make an entry in it and put DF value '0' for now
            // if(DFVector->find(Vocabulary[word])==DFVector->end())
            if(Vocabulary->find(word)==Vocabulary->end())
            {
              Vocabulary->insert(make_pair(word,vocabularyIndex));
              DFVector->insert(make_pair((*Vocabulary)[word],0));
              // std::cout << (*Vocabulary)[word] << '\n';
              vocabularyIndex++;
            }

            ////if the term is not in TFVector
            //then make an entry in it and put TF value '1'
            if(TFScores.find((*Vocabulary)[word])==TFScores.end())
            {
              TFScores.insert(make_pair((*Vocabulary)[word],1));
            }
            else //if the term is in TFVector then increment its TF value by 1
            {
              TFScores[(*Vocabulary)[word]]+=1;
            }
          }
        }
      }
    // }
  }
  map<int,float>::iterator TFVectorIterator=TFScores.begin();

  while(TFVectorIterator != TFScores.end())
  {
    //Remove low TF stems
    if(TFVectorIterator->second < 2)
    {
      TFVectorIterator=TFScores.erase(TFVectorIterator);
    }
    else //Update DF
    {
      (*DFVector)[TFVectorIterator->first]+=1;
      TFVectorIterator++;
    }
  }
  return TFScores;
}

void getTFIDFScores(map<int,float> *DFVector,map<string, map<int,float> > *TFIDFMatrix)
{
  map<string, map<int,float> >::iterator TFIDFMatrixIterator=TFIDFMatrix->begin();
  while(TFIDFMatrixIterator!=TFIDFMatrix->end())
  {
    map<int,float>::iterator TermIterator = TFIDFMatrixIterator->second.begin();
    while(TermIterator!=TFIDFMatrixIterator->second.end())
    {
      //check if the term is in unique term vector
      //if yes, then calculate TFIDF and store it in TFIDF matrix
      //if no, then remove the term from TFIDF matrix
      if(DFVector->find(TermIterator->first)!=DFVector->end())
      {
        float TF=1+log((float)TermIterator->second);
        float IDF=log((float)TFIDFMatrix->size()/(float)(*DFVector)[TermIterator->first]);
        float TFIDF=TF*IDF;

        //Storing TFIDF value to TFIDF Matrix
        TermIterator->second=TFIDF;
        TermIterator++;
      }
      else
      {
          TermIterator=(*TFIDFMatrix)[TFIDFMatrixIterator->first].erase(TermIterator);
      }
    }
    TFIDFMatrixIterator++;
  }
}

//Usage function for making the user understand the usage of the program
void usage()
{
  std::cout << "usage: ";
  std::cout << "TFIDF_Vectorizer documents_directory stopwords_file TFIDF_output_file" << '\n';
  std::cout << "Arguments:" << '\n';
  std::cout << "    documents_directory   : the documents directory(Include the path as well if in other than root directory)" << '\n';
  std::cout << "    stopwords_file        : the stopwords file(Include the path as well if in other than root directory" << '\n';
  std::cout << "    TFIDF_output_file     : the output file(Include the path as well if in other than root directory" << '\n';
}

int main(int argc, char** argv)
{
  //Checking for the arguments
  if(argc<4 || (argc=2 && argv[1]=="-h"))
  {
    usage();
    return -1;
  }

  map<int,float> DFVector;
  map<string,int> Vocabulary;
  map<string, map<int,float> > TFIDFMatrix;
  std::string dir=argv[1];
  string stopfile=argv[2];
  string output_file=argv[3];
  std::string file_content;
  ofstream outfile,cluster_ground_truth_file,vocabularyfile;
  clock_t begin,end,prog_start,prog_end;

  prog_start=clock();

  begin = clock();
  std::cout << "Reading the stopwords...." << '\n';
  read_stopwords(stopfile);
  end = clock();
  std::cout << "Stopwords reading done in : " << float(end - begin) / CLOCKS_PER_SEC << "\n\n";

  std::cout << "Iterating through the documents for calculating TF Scores...." << '\n';
  begin=clock();
  //Iterating through the directory specified
  int cluster_count=(-1);
  int file_count=(-1);
  cluster_ground_truth_file.open("cluster_ground_truth.txt");
  for(auto& item: fs::recursive_directory_iterator(dir))
  {
    //checking if it is a regular file
    //if yes, then open the file
    if(fs::is_regular_file(item.path()))
    {
      file_count++;
      cluster_ground_truth_file << file_count << ":" << cluster_count << ":" << item << '\n';
      string file_name=item.path().string();
      std::ifstream infile(file_name);

      //File Opening Error Handling
      if(!infile)
      {
        std::cout << "Can not open infile.\n";
        return 1;
      }

      //storing the entire file content in a string
      file_content.assign((std::istreambuf_iterator<char>(infile)),(std::istreambuf_iterator<char>()));

      //Converting to lower case
      std::transform(file_content.begin(), file_content.end(), file_content.begin(), ::tolower);

      //calculating TF Scores for this document
      map<int,float> TFVector=getTFScores(file_content,&DFVector,&Vocabulary);

      //Inserting the pair of file name & TF Vector into TFIDF Matrix
      TFIDFMatrix.insert(make_pair(file_name,TFVector));
      TFVector.clear();
      infile.close();
    }
    else
    {
      cluster_count++;
    }
    file_content.clear();
  }
  end=clock();
  std::cout << "TF Scores Calculation Done in : "<< float(end - begin) / CLOCKS_PER_SEC << "\n\n";

  std::cout << "Calculating TFIDF Scores...." << '\n';
  begin=clock();

  //Calculating TFIDF Scores for each entry in TFIDF Matrix
  getTFIDFScores(&DFVector,&TFIDFMatrix);
  end=clock();
  std::cout << "TFIDF Scores Calculation Done in : "<< float(end - begin) / CLOCKS_PER_SEC << "\n\n";

  std::cout << "DF Matrix Size(Unique Terms) : " << DFVector.size() << '\n';
  std::cout << "TFIDF Matrix Size : " << TFIDFMatrix.size() << "\n\n";

  std::cout << "Writing to output file...." << '\n';
  begin=clock();
  outfile.open(output_file);
  vocabularyfile.open("./vocab.txt");
  for(map<string,int>::iterator vocabIterator = Vocabulary.begin();vocabIterator!=Vocabulary.end();vocabIterator++)
  {
    vocabularyfile << vocabIterator->first << " : " << vocabIterator->second << " \n";
  }
  map<string, map<int,float> >::iterator TFIDFMatrixIterator=TFIDFMatrix.begin();

  // while(TFIDFMatrixIterator!=TFIDFMatrix.end())
  // {
  //   map<std::string,float>::iterator UniqueTermIterator = DFVector.begin();
  //   map<std::string,float>::iterator TermIterator = TFIDFMatrixIterator->second.begin();
  //   while(UniqueTermIterator!=DFVector.end())
  //   {
  //     if(TFIDFMatrixIterator->second.find(UniqueTermIterator->first)!=TFIDFMatrixIterator->second.end())
  //     {
  //       outfile << TFIDFMatrixIterator->second[UniqueTermIterator->first] << ",";
  //     }
  //     else
  //     {
  //       outfile << float(0) << ",";
  //     }
  //
  //     UniqueTermIterator++;
  //   }
  //   outfile << "\n";
  //   TFIDFMatrixIterator++;
  // }

  outfile << DFVector.size() << "\n";
  while(TFIDFMatrixIterator!=TFIDFMatrix.end())
  {
    map<int,float>::iterator TermIterator = TFIDFMatrixIterator->second.begin();
    // map<int,std::string>::iterator vocabularyIterator = DFVector.begin()
    while(TermIterator!=TFIDFMatrixIterator->second.end())
    {
      // outfile << Vocabulary.find(TermIterator->first) << "-" << TFIDFMatrixIterator->second[TermIterator->first] << ",";
      outfile << TermIterator->first << ":" << TermIterator->second << ",";
      // TFIDFMatrixIterator->second[TermIterator->first] << ",";

      TermIterator++;
    }
    outfile << "\n";
    TFIDFMatrixIterator++;
  }

  end = clock();
  std::cout << "time for writing to file: " << float(end - begin) / CLOCKS_PER_SEC << "\n\n";
  prog_end=clock();
  std::cout << "Total time taken by the program : "<< float(prog_end - prog_start) / CLOCKS_PER_SEC << "\n\n";
  return 0;
}
