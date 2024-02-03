import os
from pprint import pprint

def read_lists_from_file_separate(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    lines = content.splitlines()

    lists = []
    current_list = []

    for line in lines:
        if line.strip() == '':
            if current_list:  # append only non-empty lists
                lists.append(current_list)
            current_list = []
        else:
            current_list.append(line)

    # append the last list if it's non-empty
    if current_list:
        lists.append(current_list)

    return lists


def read_lists_from_file(file_path, mode='separate'):
    # list of list of strings, sublists are separated by empty lines in the file
    list_of_list = read_lists_from_file_separate(file_path) 
    if mode == 'separate': # return original list of list
      return list_of_list
    elif mode == 'one': # return the concatenation of all sublists
      return [item for sublist in list_of_list for item in sublist]
    elif type(mode) == dict: # more detailed control
      return_list = []
      for i in range(len(list_of_list)):
        sublist = list_of_list[i] # list of strings
        indices = eval(mode[str(i)]) # list of indices
        print(i, indices)
        add_list = []
        for j in indices:
          try:
            add_list.append(sublist[j])
          except IndexError:
            print('IndexError: index {} out of range for list {} of length {}'.format(j, i, len(sublist)))
        return_list.extend(add_list)
      return return_list 
    else:
      raise ValueError('mode must be "separate", "one" or a dict')

def read_whole_file(file_path):
  with open(file_path, 'r') as file:
    content = file.read()
  return content

def write_whole_file(file_path, content):
  with open(file_path, 'w') as file:
    file.write(content)



if __name__ == "__main__":
  group1, group2, group3 = read_lists_from_file('captions/ode1.md')
  for g in [group1, group2, group3]:
    print('---')
    for line in g:
      print(line)