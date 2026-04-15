#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Elena Alafuzova + George Kordonis

"""
import os
import xml.etree.ElementTree as ET

tree = ET.parse('ABSA16_Restaurants_Train_SB1_v2.xml')
root = tree.getroot()

total_instances = len(root.findall('.//Review'))  #  each instance is a direct child of the root element

num_parts = 10
part_size = total_instances // num_parts

# split XML data into parts
xml_parts = []
start_index = 0

# itirate over instances and split them into separate parts, then store them in the list above
for i in range(num_parts):
    end_index = start_index + part_size
    part = ET.Element('Reviews')  # Create a new root element for each part
    part.extend(root.findall('.//Review')[start_index:end_index])  # Append Review elements to the new root element
    xml_parts.append(part)
    start_index = end_index

# handle any remaining instances if the division is not exact
if start_index < total_instances:
    remaining_instances = ET.Element('Reviews')
    remaining_instances.extend(root.findall('.//Review')[start_index:])
    xml_parts.append(remaining_instances)

# process each part or save it to a separate XML file into the new folder "data"
output_directory = "./data" #Specify the path to the folder

# create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

for i, part in enumerate(xml_parts):
    part_filename = os.path.join(output_directory, f'part_{i+1}.xml')
    part_tree = ET.ElementTree(part)
    part_tree.write(part_filename)


# print the number of instances in the original datafile
print(f"Number of reviews in the original data file: {total_instances}")
print()
# print the number of files in the new directory
file_count = len([filename for filename in os.listdir(output_directory) if filename.endswith('.xml')])
print(f"Number of files in the directory with the splitted data: {file_count}")
print()
# print the number of instances in each file
print("Number of reviews in the splitted data:")
for i in range(file_count):
    file_name = f'part_{i+1}.xml'
    file_path = os.path.join(output_directory, f'part_{i+1}.xml')
    tree = ET.parse(file_path)
    root = tree.getroot()
    instance_count = len(root.findall('.//Review'))
    print(f"{file_name}: Number of reviews: {instance_count}")