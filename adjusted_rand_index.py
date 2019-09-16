from sklearn.metrics.cluster import adjusted_rand_score
import sys

truth_txt=int(sys.argv[1])
# members = str(sys.argv[1])
ground_truth_file = open("cluster_ground_truth.txt","r",encoding = "ISO-8859-1").read()
true_clusters = ground_truth_file.splitlines()

members_file = open(truth_txt,"r",encoding = "ISO-8859-1").read()
members = members_file.splitlines()

ground_truth=[]
prediction=[]

for membership in members:
    cluster_doc_pair=membership.strip().split(":")
    doc=int(cluster_doc_pair[0].strip())
    cluster=int(cluster_doc_pair[1].strip())
    prediction.append(cluster)
    ground_truth.append(int(true_clusters[doc].split(":")[1].strip()))

score = adjusted_rand_score(ground_truth,prediction)
print("adjusted_rand_score = "+str(score))



#
# clusters_file = open("clusters_try.txt","r",encoding = "ISO-8859-1").read()
# ground_truth_file = open("cluster_ground_truth.txt","r",encoding = "ISO-8859-1").read()
#
# predicted_clusters = clusters_file.splitlines()
# true_clusters = ground_truth_file.splitlines()
#
# total_clusters = len(predicted_clusters)
# score=0.0
# for cluster in predicted_clusters:
#     ground_truth=[]
#     prediction=[]
#     cluster_key_values=cluster.split(":")
#     all_docs=cluster_key_values[1].strip()
#     cluster_docs=all_docs.split(",")
#     cluster_docs=cluster_docs[:(len(cluster_docs)-1)]
#     for doc in cluster_docs:
#         prediction.append(int(cluster_key_values[0].strip()))
#         ground_truth.append(int(true_clusters[int(doc)].split(":")[1]))
#
#
# print("adjusted_rand_score = "+str(score/total_clusters))
