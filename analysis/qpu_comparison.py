# type: ignore
# Copyright (C) 2024 AljaÅ¾ Krpan <krpan.aljaz@gmail.com>
 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
 
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>. 

##########################################################################

import pickle
from pathlib import Path
import os
import networkx as nx
from tqdm import tqdm
import math
import numpy as np
from time import time

from collections import defaultdict
from dimod import SampleSet
from dimod import ExactSolver
from dwave.samplers import SimulatedAnnealingSampler

def solution_correction(
        stable_set_graph,
        solution_nodes
    ):
    subgraph = nx.Graph(stable_set_graph.subgraph(solution_nodes))
    # we go through each vertex, and if it has any edges connected to it, we remove it.
    for n in list(subgraph.nodes):
        if(subgraph.degree(n) > 0):
            subgraph.remove_node(n)
    return list(subgraph.nodes)

def calculate_best_solution(
        stable_set_graph,
        sampler,
        beta=0.5,
        num_of_runs=1,
        num_of_part=1,
        output_file=None,
        no_output_file = False,
        console_output=True,
        partition_with_halo_list=None
    ):
    # if we want to produce a file but the directory does not exist or is not given, throw an Exception,
    # otherwise we might waste computational resources and be left without the result
    if(not no_output_file and (output_file is None or not os.path.exists(os.path.dirname(output_file)))):
        raise Exception("Output directory not given or does not exist! If you don't want to produce the output file, set no_output_file=True in function call.")
    #If output directory already contains a file with the same name, throw an Exception, otherwise we might overwrite the file
    if(not no_output_file and os.path.exists(output_file)):
        raise Exception("Output file already exists! Delet the file or if you don't want to produce the output file, set no_output_file=True in function call.")
    if(False):
        print(f"Using {sampler} sampler")
    # this has to be done, otherwise the partition function with input num_of_part=1 throws an error
    if(num_of_part > 1 and partition_with_halo_list is None):
        raise Exception("Partition number is greater than 1, but partition_with_halo_list is not given!")
    elif(partition_with_halo_list is None):
        partition_with_halo_list = [set(stable_set_graph.nodes)]

    best_solution_nodes = []
    best_solution_energy = 0
    all_solutions_info = {}

    # we go through every partition, we create subgraph and Q matrix (which represents QUBO function)
    # we calculate the response, which is saved in a file (if desired), we save the best solution nodes
    # and energy, and we print the best result
    for k, partition in enumerate(partition_with_halo_list):
        if(len(partition_with_halo_list) > 1):
            subgraph = stable_set_graph.subgraph(list(partition))
        else:
            subgraph = stable_set_graph

        Q = defaultdict(int)
        for i in partition:
            Q[(i,i)]+= -1
        for i, j in subgraph.edges:
            # because we'll be using upper triangular adjacency matrix, we modify beta by multiply it by 2
            Q[(i,j)]+= beta*2

        
        if(num_of_runs > 1):
            response = sampler.sample_qubo(Q, num_reads=num_of_runs)
        else:
            response = sampler.sample_qubo(Q)

        if(not no_output_file):
            if(len(partition_with_halo_list) > 1):
                with open(output_file+"_partition"+str(k).zfill(int(math.log10(len(partition_with_halo_list)-1)+1))+".pkl", 'wb') as file:
                    pickle.dump(response.to_serializable(), file)
            else:
                with open(output_file, 'wb') as file:
                    pickle.dump(response.to_serializable(), file)

        # this is how you get solution nodes and solution energy from the response
        solution_nodes = [h for h,v in response.first[0].items() if v == 1]
        corrected_solution_nodes = solution_correction(stable_set_graph, solution_nodes)
        solution_energy = response.first[1]

        all_solutions_info[k] = {
            "best_solution_nodes" : solution_nodes,
            "best_solution_energy" : solution_energy,
            "sample_set": response
        }
        
        if(solution_energy < best_solution_energy):
            best_solution_nodes = solution_nodes
            best_solution_energy = solution_energy
        
        # this is to avoid double printing the result if we choose to calculate response without partitions
        if(len(partition_with_halo_list) > 1 and console_output):
            k_pad = f"{k+1:>{int(math.log(len(partition_with_halo_list),10))+1}}"
            print(f"Partition {k_pad}: beta={beta} || solution energy: {solution_energy} || number of edges in solution: {len(subgraph.subgraph(solution_nodes).edges)} || solution size: {len(solution_nodes)}")

    if(console_output):
        print(f"Best result: beta={beta} || best solution energy: {best_solution_energy} || number of edges in best solution: {len(subgraph.subgraph(best_solution_nodes).edges)} || best solution size: {len(best_solution_nodes)}\n")

    return all_solutions_info

def open_saved_sample_set(input_file):
    with open(input_file, 'rb') as file:
        sample_set = SampleSet.from_serializable(pickle.load(file))
    return sample_set

# Function takes a stable set graph and return the annihilation number of the graph.
def annihilation_number(stable_set_graph):
    edge_number = len(stable_set_graph.edges)
    degrees = list(stable_set_graph.degree())
    degrees.sort(key=lambda x: x[1])
    sum = 0
    for i in range(len(degrees)):
        sum += degrees[i][1]
        if(sum > edge_number):
            return i
    return len(degrees)

# Function takes a stable set graph on which samples were calculated, a sample set, a beta value on which samples were calculated.
# Optionally, you can set the sampler and number of runs for the recalculation of the solutions.
# The function returns a dict of the best recalculated solution and all recalculated solutions.
def eliminate_and_recalculate(
        stable_set_graph,
        sample_set,
        sample_beta,
        sampler=SimulatedAnnealingSampler(),
        num_of_runs=100,
        console_output=True
    ):

    recalc_beta = max(0.5, sample_beta)

    first_solution = [k for k,v in sample_set.first[0].items() if v == 1]
    edges_in_first_solution = len(stable_set_graph.subgraph(first_solution).edges)
    best = len(first_solution) - edges_in_first_solution

    sorted_solutions = sample_set.record.tolist()
    # Sort by cost function
    sorted_solutions.sort(key=lambda x : x[1])

    solutions_to_recalc = []
    #for record in sample_set.record:
    for record in sorted_solutions:
        solutions_to_recalc.append([sample_set.variables[k[0]] for k,v in np.ndenumerate(record[0]) if v == 1])
    #solutions_to_recalc = sorted(solutions_to_recalc, key=lambda x: len(x) - 2*sample_beta*len(stable_set_graph.subgraph(x).edges), reverse=True)

    recalculated_solutions = []
    num_of_recalculated_solutions = 0
    max_component_size = 0
    for solution in tqdm(solutions_to_recalc, disable=not console_output):
        stable_set_subgraph = stable_set_graph.subgraph(solution)
        if(annihilation_number(stable_set_subgraph) > best):
            num_of_recalculated_solutions += 1

            components = nx.connected_components(stable_set_subgraph)
            new_solution = set()
            for i in components:
                if len(i) > max_component_size:
                    max_component_size = len(i)
                if len(i) == 1:
                    new_solution = new_solution.union(i)
                else:
                    component_subgraph = stable_set_subgraph.subgraph(i)
                    response = calculate_best_solution(component_subgraph, sampler, num_of_runs=num_of_runs, no_output_file=True, console_output=False, beta=recalc_beta)
                    corrected_solution = solution_correction(component_subgraph, response[0]["best_solution_nodes"])
                    new_solution = new_solution.union(set(corrected_solution))

            if(len(new_solution) > best):
                best = len(new_solution)
            recalculated_solutions.append({"recalculated_solution_nodes": new_solution,
                                        "recalcualted_solution_energy": (-1)*len(new_solution)})
    
    num_of_recalculated_solutions = len(recalculated_solutions)
    if(num_of_recalculated_solutions == 0):
        new_solution = set(solution_correction(stable_set_graph, first_solution))
        recalculated_solutions.append({"recalculated_solution_nodes": new_solution,
                                        "recalcualted_solution_energy": (-1)*len(new_solution)})

    #print("Max Component size ", max_component_size)
    return {"best_recalculated_solution": min(recalculated_solutions, key= lambda x : x["recalcualted_solution_energy"]),
            "all_recalculated_solutions": recalculated_solutions,
            "num_of_recalculated_solutions": num_of_recalculated_solutions,
            "max_component_size": max_component_size}