import networkx as nx
import random
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.pyplot import show
from ipywidgets import GridBox, Layout, interactive_output
from love import calculateODEMatrixVector
from IPython.display import clear_output
from IPython.display import display
import pickle
random.seed(0)
num = 0

def random_starting_matrix():
    A = np.ones((4, 4))
    for i in range(4):
        for j in range(4):
            A[i, j] = round(random.uniform(-1, 1), 1)
    return A

def random_starting_vector():
    A = np.ones((4))
    for i in range(4):
        A[i] = round(random.uniform(-1, 1), 1)
    return A

def xy_starting_vector():
    A = np.zeros((4))
    return A


number_of_nodes = 10
G = nx.complete_graph(number_of_nodes)
nx.set_edge_attributes(G, [], "t")
nx.set_edge_attributes(G, [], "data")
nx.set_edge_attributes(G, [], "parameters")
out = widgets.Output()

#initialize each node with the roamntic parameters (matrix and vector) for a different node
for i in range(number_of_nodes):
    for j in range(i+1, number_of_nodes):
        G[i][j]['parameters'] = [random_starting_matrix(), random_starting_vector(), xy_starting_vector()]

for i in range(number_of_nodes):
    for j in range(i+1, number_of_nodes):
        calculatedODE = calculateODEMatrixVector(G[i][j]['parameters'][0], G[i][j]['parameters'][1], G[i][j]['parameters'][2])
        G[i][j]["data"] = calculatedODE.y
        G[i][j]["t"] = calculatedODE.t
        
        
slider_style = {'description_width': 'initial'} 
slider_layout = Layout(width='auto')
# Parameters

#initial starting parameters for graph
# Matrix A
# G[0][1]['parameters'][0][0][0] means select edge from node 0 to node 1, select the parameter attribute, matrix A, row 0, column 0
axx= G[0][1]['parameters'][0][0][0]; axy= G[0][1]['parameters'][0][0][1]; bxx= G[0][1]['parameters'][0][0][2]; bxy= G[0][1]['parameters'][0][0][3]
ayx= G[0][1]['parameters'][0][1][0]; ayy= G[0][1]['parameters'][0][1][1]; byx= G[0][1]['parameters'][0][1][2]; byy= G[0][1]['parameters'][0][1][3]
cxx= G[0][1]['parameters'][0][2][0]; cxy= G[0][1]['parameters'][0][2][1]; dxx= G[0][1]['parameters'][0][2][2]; dxy= G[0][1]['parameters'][0][2][3]
cyx= G[0][1]['parameters'][0][3][0]; cyy= G[0][1]['parameters'][0][3][1]; dyx= G[0][1]['parameters'][0][3][2]; dyy= G[0][1]['parameters'][0][3][3]
#Vector B
# G[0][1]['parameters'][1][0] means select eedge from node 0 to node 1, select the parameter attirbute, vector B, row 0
fxy=G[0][1]['parameters'][1][0]; fyx=G[0][1]['parameters'][1][1]; gxy=G[0][1]['parameters'][1][2]; gyx=G[0][1]['parameters'][1][3]
# Vector C
# G[0][1]['parameters'][2][0] means select edge form node 0 to node 1, select the parameter attribute, vector C, row 0
xi0=G[0][1]['parameters'][2][0]; yi0=G[0][1]['parameters'][2][1]; xp0=G[0][1]['parameters'][2][2]; yp0=G[0][1]['parameters'][2][3]

node1 = widgets.Dropdown(options=[str(i) for i in range(number_of_nodes)], value='0', description='Person:')
node2 = widgets.Dropdown(options=[str(i) for i in range(number_of_nodes)], value='1', description='Person:') 

axx_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=axx, description='axx', style=slider_style, layout=slider_layout)
axy_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=axy, description='axy', style=slider_style, layout=slider_layout)
bxx_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=bxx, description='bxx', style=slider_style, layout=slider_layout)
bxy_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=bxy, description='bxy', style=slider_style, layout=slider_layout)
ayx_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=ayx, description='ayx', style=slider_style, layout=slider_layout)
ayy_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=ayy, description='ayy', style=slider_style, layout=slider_layout)
byx_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=byx, description='byx', style=slider_style, layout=slider_layout)
byy_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=byy, description='byy', style=slider_style, layout=slider_layout)
cxx_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=cxx, description='cxx', style=slider_style, layout=slider_layout)
cxy_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=cxy, description='cxy', style=slider_style, layout=slider_layout)
dxx_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=dxx, description='dxx', style=slider_style, layout=slider_layout)
dxy_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=dxy, description='dxy', style=slider_style, layout=slider_layout)
cyx_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=cyx, description='cyx', style=slider_style, layout=slider_layout)
cyy_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=cyy, description='cyy', style=slider_style, layout=slider_layout)
dyx_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=dyx, description='dyx', style=slider_style, layout=slider_layout)
dyy_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=dyy, description='dyy', style=slider_style, layout=slider_layout)

fxy_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=fxy, description='fxy', style=slider_style, layout=slider_layout)
fyx_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=fyx, description='fyx', style=slider_style, layout=slider_layout)
gxy_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=gxy, description='gxy', style=slider_style, layout=slider_layout)
gyx_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=gyx, description='gyx', style=slider_style, layout=slider_layout)

xi0_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=xi0, description='xi0', style=slider_style, layout=slider_layout)
yi0_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=yi0, description='yi0', style=slider_style, layout=slider_layout)
xp0_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=xp0, description='xp0', style=slider_style, layout=slider_layout)
yp0_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.1, value=yp0, description='yp0', style=slider_style, layout=slider_layout)

savebutton = widgets.Button(description="Save Parameters")
randomizebutton = widgets.Button(description="Randomized Parameters")
updateplotbutton = widgets.Button(description="Update Plot (not saved)")
downloadGbutton = widgets.Button(description="save graph to file")
uploadGfile = widgets.Text(value='', placeholder='give the path of the file you want to upload')
uploadGfilebutton = widgets.Button(description="Submit file path")

normalizecheckbox = widgets.Checkbox(value=False,description='Normalize values across all connectoins with sum=0 when saving parameters')

def update_slider_values():
    node1_value = int(node1.value)
    node2_value = int(node2.value)
    axx_slider.value = G[node1_value][node2_value]['parameters'][0][0][0]
    axy_slider.value = G[node1_value][node2_value]['parameters'][0][0][1]
    bxx_slider.value = G[node1_value][node2_value]['parameters'][0][0][2]
    bxy_slider.value = G[node1_value][node2_value]['parameters'][0][0][3]
    ayx_slider.value = G[node1_value][node2_value]['parameters'][0][1][0]
    ayy_slider.value = G[node1_value][node2_value]['parameters'][0][1][1]
    byx_slider.value = G[node1_value][node2_value]['parameters'][0][1][2]
    byy_slider.value = G[node1_value][node2_value]['parameters'][0][1][3]
    cxx_slider.value = G[node1_value][node2_value]['parameters'][0][2][0]
    cxy_slider.value = G[node1_value][node2_value]['parameters'][0][2][1]
    dxx_slider.value = G[node1_value][node2_value]['parameters'][0][2][2]
    dxy_slider.value = G[node1_value][node2_value]['parameters'][0][2][3]
    cyx_slider.value = G[node1_value][node2_value]['parameters'][0][3][0]
    cyy_slider.value = G[node1_value][node2_value]['parameters'][0][3][1]
    dyx_slider.value = G[node1_value][node2_value]['parameters'][0][3][2]
    dyy_slider.value = G[node1_value][node2_value]['parameters'][0][3][3]

    fxy_slider.value = G[node1_value][node2_value]['parameters'][1][0]
    fyx_slider.value = G[node1_value][node2_value]['parameters'][1][1]
    gxy_slider.value = G[node1_value][node2_value]['parameters'][1][2]
    gyx_slider.value = G[node1_value][node2_value]['parameters'][1][3]

    xi0_slider.value = G[node1_value][node2_value]['parameters'][2][0]
    yi0_slider.value = G[node1_value][node2_value]['parameters'][2][1]
    xp0_slider.value = G[node1_value][node2_value]['parameters'][2][2]
    yp0_slider.value = G[node1_value][node2_value]['parameters'][2][3]
    
    
def update_plot(t_data = None, y_data = None):
    with out:
        clear_output()
        node1_value = int(node1.value)
        node2_value = int(node2.value)
        if (node1_value == node2_value):
            print("Please select two different people")
            return
        if (t_data is not None and y_data is not None):
            t = t_data
            xa = y_data
        else:
            xa = G[int(node1_value)][int(node2_value)]["data"]
            t = G[int(node1_value)][int(node2_value)]["t"]
        
        # Set up a figure twice as tall as it is wide
        fig = plt.figure(figsize=(10, 10))
        # first subplot
        ax = fig.add_subplot(2, 2, 1)
        ax.scatter(xa[0], xa[1], s=100, cmap='viridis', alpha=0.5, c=t)
        ax.scatter(xa[2], xa[3], s=100, cmap='viridis', alpha=0.5, c=t)
        ax.plot(xa[0],xa[1], c="black", label="Intimacy", alpha=0.5)
        ax.plot(xa[2],xa[3], c="red", label="Passion", alpha=0.5)
        ax.set_xlabel("Person " + node1.value, fontsize=16)
        ax.set_ylabel("Person " + node2.value, fontsize=16)
        ax.grid(True)
        ax.legend()

        # second subplot
        ax = fig.add_subplot(2, 2, 2)
        ax.scatter(xa[0], xa[2], s=100, cmap='viridis', alpha=0.5, c=t)
        ax.scatter(xa[1], xa[3], s=100, cmap='viridis', alpha=0.5, c=t)
        ax.plot(xa[0],xa[2], c="black", label="Person " + node2.value, alpha=0.5)
        ax.plot(xa[1],xa[3], c="red", label="Person " + node1.value, alpha=0.5)
        ax.set_xlabel('Intimacy', fontsize=16)
        ax.set_ylabel('Passion', fontsize=16)

        ax.grid(True)
        ax.legend()

        # third subplot
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.scatter(xa[0], xa[1], xa[2], c=t, s=100, cmap='viridis', alpha=0.5)
        ax.set_xlabel('Intimacy of Person ' + node1.value, fontsize=16)
        ax.set_ylabel('Intimacy of Person ' + node2.value, fontsize=16)
        ax.set_zlabel('Passion of Person ' + node1.value, fontsize=16)
        ax.grid(True)

        # fourth subplot
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.scatter(xa[0], xa[1], xa[3], c=t, cmap='viridis', alpha=0.5, s=100)
        ax.set_xlabel('Intimacy of Person' + node1.value, fontsize=16)
        ax.set_ylabel('Intimacy of Person ' + node2.value, fontsize=16)
        ax.set_zlabel('Passion of Person ' + node2.value, fontsize=16)
        ax.grid(True)
        plt.show()
        show()

def normalize_values():
    matrixA = [
        [[], [], [], []], 
        [[], [], [], []], 
        [[], [], [], []], 
        [[], [], [], []]
    ]
    vectorB = [[], [], [], []]
    for i, j, d in G.edges(data=True):
        a = d['parameters'][0]
        b = d['parameters'][1]
        for row in range(len(a)):
            for col in range(len(a[0])):
                matrixA[row][col].append(a[row][col])
        for row in range(len(b)):
            vectorB[row].append(b[row])
    
    for row in range(len(matrixA)):
        for col in range(len(matrixA[0])):
            norm = np.array(matrixA[row][col])/np.linalg.norm(matrixA[row][col])
            final_vector = norm - np.mean(norm)
            matrixA[row][col] = final_vector
    for row in range(len(vectorB)):
        norm = np.array(vectorB[row])/np.linalg.norm(vectorB[row])
        final_vector = norm - np.mean(norm)
        vectorB[row] = final_vector

    counter = 0
    for i, j, d in G.edges(data=True):
        matrixa = np.array([[matrixA[0][0][counter], matrixA[0][1][counter], matrixA[0][2][counter], matrixA[0][3][counter]], [matrixA[1][0][counter], matrixA[1][1][counter], matrixA[1][2][counter], matrixA[1][3][counter]], [matrixA[2][0][counter], matrixA[2][1][counter], matrixA[2][2][counter], matrixA[2][3][counter]], [matrixA[3][0][counter], matrixA[3][1][counter], matrixA[3][2][counter], matrixA[3][3][counter]]])
        G[i][j]['parameters'][0] = matrixa

        vectorb = np.array([[vectorB[0][counter]], [vectorB[1][counter]], [vectorB[2][counter]], [vectorB[3][counter]]])
        G[i][j]['parameters'][1] = vectorb
        counter += 1

def save_graph(change):
    pickle.dump(G, open('love_multiple.pickle', 'wb'))

# load graph object from file
def upload_graph(change):
    global G
    G = pickle.load(open(uploadGfile.value, 'rb'))
    node1_value = int(node1.value)
    node2_value = int(node2.value)
    update_slider_values()
    update_plot()
    
dropdown = [node1, node2] 
button = [savebutton, randomizebutton, updateplotbutton, downloadGbutton]
checkbox = [normalizecheckbox, uploadGfile, uploadGfilebutton]
sliders = [axx_slider,axy_slider,bxx_slider,bxy_slider,ayx_slider,ayy_slider,byx_slider,byy_slider, cxx_slider,cxy_slider,dxx_slider,dxy_slider,cyx_slider,cyy_slider,dyx_slider,dyy_slider,fxy_slider,fyx_slider,gxy_slider,gyx_slider,xi0_slider,yi0_slider,xp0_slider,yp0_slider]

def update_sliders(change):
    node1_value = int(node1.value)
    node2_value = int(node2.value)
    if (node1_value == node2_value):
        print("Please select two different people")
        return
    
    update_slider_values()
    update_plot()

node1.observe(update_sliders, names='value')
node2.observe(update_sliders, names='value')

grid_dropdown = GridBox(dropdown, layout=Layout(
    width='100%',
    grid_template_columns='repeat(2, 30%)',  
    grid_gap='20px 20px'
))
grid_buttons = GridBox(button, layout=Layout(
    width='100%',
    grid_template_columns='repeat(5, 15%)',  
    grid_gap='20px 20px'
))
grid_checkboxes = GridBox(checkbox, layout=Layout(
    width='100%',
    grid_template_columns='repeat(1, 20%)',  
    grid_gap='20px 20px'
))
grid_sliders = GridBox(sliders, layout=Layout(
    width='100%',
    grid_template_columns='repeat(4, 20%)',  
    grid_gap='20px 20px'
))

def save_button(b):
    node1_value = int(node1.value)
    node2_value = int(node2.value)
    axx = axx_slider.value
    axy = axy_slider.value
    bxx = bxx_slider.value
    bxy = bxy_slider.value
    ayx = ayx_slider.value
    ayy = ayy_slider.value
    byx = byx_slider.value
    byy = byy_slider.value
    cxx = cxx_slider.value
    cxy = cxy_slider.value
    dxx = dxx_slider.value
    dxy = dxy_slider.value
    cyx = cyx_slider.value
    cyy = cyy_slider.value
    dyx = dyx_slider.value
    dyy = dyy_slider.value
    fxy = fxy_slider.value
    fyx = fyx_slider.value
    gxy = gxy_slider.value
    gyx = gyx_slider.value
    xi0 = xi0_slider.value
    yi0 = yi0_slider.value
    xp0 = xp0_slider.value
    yp0 = yp0_slider.value

    if (node1_value == node2_value):
        print("Please select two different people")
        return

    G[int(node1_value)][int(node2_value)]['parameters'][0] = np.array([[axx, axy, bxx, bxy], [ayx, ayy, byx, byy], [cxx, cxy, dxx, dxy], [cyx, cyy, dyx, dyy]])
    G[int(node1_value)][int(node2_value)]['parameters'][1] = np.array([fxy, fyx, gxy, gyx])
    G[int(node1_value)][int(node2_value)]['parameters'][2] = np.array([xi0, yi0, xp0, yp0])

    if (normalizecheckbox.value):
        normalize_values()

    calculatedODE = calculateODEMatrixVector(G[node1_value][node2_value]['parameters'][0], G[node1_value][node2_value]['parameters'][1], G[node1_value][node2_value]['parameters'][2])
    G[node1_value][node2_value]["data"] = calculatedODE.y
    G[node1_value][node2_value]["t"] = calculatedODE.t
    update_plot()

def randomize_button(b):
    node1_value = int(node1.value)
    node2_value = int(node2.value)
    G[node1_value][node2_value]['parameters'][0] = random_starting_matrix()
    G[node1_value][node2_value]['parameters'][1] = random_starting_vector()
    G[node1_value][node2_value]['parameters'][2] = xy_starting_vector()

    calculatedODE = calculateODEMatrixVector(G[node1_value][node2_value]['parameters'][0], G[node1_value][node2_value]['parameters'][1], G[node1_value][node2_value]['parameters'][2])
    G[node1_value][node2_value]["data"] = calculatedODE.y
    G[node1_value][node2_value]["t"] = calculatedODE.t

    update_slider_values()

    update_plot()
def update_plot_button(b):
    matrixA = np.array([[axx_slider.value, axy_slider.value, bxx_slider.value, bxy_slider.value], [ayx_slider.value, ayy_slider.value, byx_slider.value, byy_slider.value], [cxx_slider.value, cxy_slider.value, dxx_slider.value, dxy_slider.value], [cyx_slider.value, cyy_slider.value, dyx_slider.value, dyy_slider.value]])
    vectorB = np.array([fxy_slider.value, fyx_slider.value, gxy_slider.value, gyx_slider.value])
    vectorC = np.array([xi0_slider.value, yi0_slider.value, xp0_slider.value, yp0_slider.value])

    calculatedODE = calculateODEMatrixVector(matrixA, vectorB, vectorC)
    y_data  = calculatedODE.y
    t_data = calculatedODE.t

    update_plot(t_data=t_data, y_data=y_data)

savebutton.on_click(save_button)
randomizebutton.on_click(randomize_button)
updateplotbutton.on_click(update_plot_button)
downloadGbutton.on_click(save_graph)
# uploadGfile.observe(upload_graph, names='value')
uploadGfilebutton.on_click(upload_graph)

display(widgets.VBox([grid_dropdown, grid_buttons, grid_checkboxes, grid_sliders]))
display(out)

'''
how to effectively use this plot. 
1. select two people from the dropdown menu. By default, the first two people are selected (perosn 0 and 1). 
2. adjust the sliders to change the parameters of the differential equation. You can then either save the parameters to the graph object by clicking the "Save Parameters" button, or you can check the results of the parameters by clicking the "Update Plot (not saved)" button. AS the name suggests, the "Update Plot (not saved)" button does not save the parameters to the graph object. 
2a. If you click the "Save Parameters" button while the "Normalize values across all connections with sum=0" button, the parameters aross the whole graph will be normalized with the condition that the sum of the normalized values is (near) 0. 
2b. TO best use this normalizing tool, set the parameters 
3. You can randomize parameter vlaues by clicking the "Randomized Parameters" button.
4. You can save the graph object to a file (named as "love_multiple.pickle" by clicking the "save graph to file" button.
5. You can upload a graph object from a file by typing the file path in the text box (either a relative path from where this file is being run from, or by the full path if the file is stored elsehwere). Then click on the "Submit file path" button. The data is then loaded into the graph object.
'''
