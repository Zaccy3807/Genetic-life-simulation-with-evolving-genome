import numpy as np
import tkinter as tk

#Input node, output node, recurrent node. hidden node, red, green, blue, energy (max), Speed, strength, Size
proteins = ['K', 'M', 'P', 'H', 'R', 'G', 'B', 'E', 'S', 'X', 'L']
#X, Y, x momentum, y momentum, angle
inputs = ['X', 'Y', 'E', 'A']
#Forward, Back, Turn Left, Turn Right, Reproduce, Attack
outputs = ['F', 'B', 'TL', 'TR', 'S']
genome_length = 1750
gene_length = 10
weight_bit = 20000
max_connections = 50
energy_decay_ps = 2.5
population_size = 200
canvas_width, canvas_height = 5000, 5000
food_per_state = 800
mutation_chance = 0.1
mutation_changes = 10
max_population = 400
energy_per_food = 25

def fromRGB(rgb):
    r, g, b = rgb
    return f'#{int(r):02x}{int(g):02x}{int(b):02x}'

def generate_genome(genome_length, gene_length):
    genome = ''
    for i in range(genome_length):
        genome += np.random.choice(proteins, p=[0.03087603305636364, 0.03887603305636364, 0.03887603305636364, 0.03087603305636364, 0.12721369539636363, 0.12721369539636363, 0.12721369539636363, 0.12721369539636363, 0.11721369539636363, 0.11721369539636363, 0.11721369539636363])
        if (i+1) % gene_length == 0:
            genome += ' '
    return genome

def decode_genome(genome):
    decoded = ''
    counts = np.zeros(len(proteins))
    for gene in genome.split():
        x = []
        for i in proteins:
            x.append(gene.count(i))
        counts[np.argmax(x)] += 1
    for P in proteins:
        decoded += f'{P}={int(counts[proteins.index(P)])}|'
    return decoded

def generate_neural_genome(nodes, genes):
    genome = ''
    for i in range(genes):
        gene = np.random.choice(nodes) + '.' + np.random.choice(nodes) + '|' + hex(int(np.random.choice(['-', '']) + ''.join([str(np.random.randint(0, 2)) for i in range(16)]), 2))
        genome += gene + ' ' if i != genes-1 else gene
    return genome

def decode_neural_genome(genome):
    connections = []
    weights = []
    for i in genome.split():
        connections.append(i.split('|')[0])
        weights.append(int(i.split('|')[1], 16)/weight_bit)
    return list(zip(connections, weights))

def mutate_genome(genome):
    if np.random.randint(1/mutation_chance+1) == 0:
        for i in range(mutation_changes):
            while True:
                index = np.random.choice(list(range(len(genome))))
                if genome[index] != ' ':
                    break
            genome = [i for i in genome]
            genome[index] = np.random.choice(proteins)
            genome = ''.join(genome)
    return genome

def mutate_inputs(inputs_):
    if len(inputs_) > 0:
        if np.random.randint(1/mutation_chance+1) == 0:
            for i in range(mutation_changes):
                inputs_[np.random.choice(list(range(len(inputs_))))] = np.random.choice(inputs)
    return inputs_

def mutate_outputs(outputs_):
    if len(outputs_) > 0: 
        if np.random.randint(1/mutation_chance+1) == 0:
            for i in range(mutation_changes):
                outputs_[np.random.choice(list(range(len(outputs_))))] = np.random.choice(outputs_)
    return outputs_

def mutate_brain(brain, all_nodes):
    if np.random.randint(1/mutation_chance+1) == 0:
        for i in range(mutation_changes):
            mutation = np.random.choice(['N1', 'N2', 'W'])
            if mutation == 'W':
                I = np.random.choice(list(range(len(brain.split()))))
                binary = bin(int(brain.split()[I].split('|')[1], 16))[3:]
                b = [n for n in binary]
                b[np.random.randint(len(b))] = np.random.choice(['0', '1'])
                hex_ = hex(int(''.join(b), 2))
                brain = brain.split()
                brain[I] = brain[I].split('|')[0] + '|' + hex_
                brain = ' '.join(brain)
            elif mutation == 'N1':
                I = np.random.choice(list(range(len(brain.split()))))
                brain = brain.split()
                N = np.random.choice(all_nodes)
                brain[I] = N + '.' + brain[I].split('.')[1]
                brain = ' '.join(brain)
            elif mutation == 'N2':
                I = np.random.choice(list(range(len(brain.split()))))
                brain = brain.split()
                N = np.random.choice(all_nodes)
                brain[I] = brain[I].split('.')[0] + '.' + N + '|' + brain[I].split('|')[1]
                brain = ' '.join(brain)
    return brain

class Cell:
    def __init__(self, genome_length=genome_length, gene_length=gene_length, parents=[]):
        if parents == []:
            self.genome = generate_genome(genome_length, gene_length)
        else:
            self.genome = mutate_genome(parents[0].genome)
            self.input_meanings = mutate_inputs(parents[0].input_meanings)
            self.output_meanings = mutate_outputs(parents[0].output_meanings)

        self.decoded_genome = decode_genome(self.genome)
        self.protein_structure = {}
        for i in range(len(proteins)):
            self.protein_structure.update({proteins[i]: int(self.decoded_genome.split('|')[i].split('=')[1])})
        
        self.color = fromRGB(np.clip([self.protein_structure['R']*7.5, self.protein_structure['G']*7.5, self.protein_structure['B']*8], 0, 255))
        self.maxEnergy = self.protein_structure['E']*3
        self.speed = self.protein_structure['S']/10
        self.strength = self.protein_structure['X']
        self.size = self.protein_structure['L']/3
        self.output_nodes = self.protein_structure['M']
        self.input_nodes = self.protein_structure['K']
        self.hidden_nodes = self.protein_structure['H'] + self.protein_structure['P']

        self.nodes = ['I'+str(i) for i in range(self.protein_structure['K'])] + ['H'+str(i) for i in range(self.protein_structure['H'])] + ['R'+str(i) for i in range(self.protein_structure['P'])] + ['O'+str(i) for i in range(self.protein_structure['M'])]
        p = self.output_nodes * self.input_nodes + self.hidden_nodes
        self.connections = p if p <= max_connections else max_connections
        if parents == []:
            self.brain = generate_neural_genome(self.nodes, self.connections)
            self.input_meanings = [np.random.choice(inputs) for i in range(self.input_nodes)]
            self.output_meanings = [np.random.choice(outputs) for i in range(self.output_nodes)]
        if parents != []:
            self.input_meanings += [np.random.choice(inputs) for i in range(self.input_nodes-len(self.input_meanings))]
            self.output_meanings += [np.random.choice(outputs) for i in range(self.output_nodes-len(self.output_meanings))]
            self.brain = mutate_brain(parents[0].brain, self.nodes)
        self.connections = decode_neural_genome(self.brain)
        for i in self.connections:
            c1, c2 = i[0].split('.')
            if c1 not in self.nodes or c2 not in self.nodes:
                self.brain = self.brain.split()
                self.brain.remove(i[0]+'|'+hex(i[1]*weight_bit))
                self.brain = ' '.join(list(set(self.brain)))
                self.connections.remove(i)
        self.recurrent_values = [0 for i in range(self.protein_structure['P'])]

    def organize(self, x):
        return x
    
    def normalize(self, data):
        data = np.array(data)
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = ((data - min_val)+1e-5) / ((max_val - min_val)+1e-5)
        return normalized_data.tolist()
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward_prop(self, x):
        if x != []:
            x = self.normalize(x)
        key = ['I', 'H', 'R', 'O']
        hidden_values = [0 for i in range(self.hidden_nodes-self.protein_structure['P'])]
        output_values = [0 for i in range(self.output_nodes)]
        values = [x, hidden_values, self.recurrent_values, output_values]
        for c, w in self.organize(self.connections):
            c1, c2 = c.split('|')[0].split('.')
            c1I = int(c1[1:])
            c2I = int(c2[1:])
            values[key.index(c2[0])][c2I] = np.tanh(values[key.index(c1[0])][c1I] * w + values[key.index(c2[0])][c2I])
        return output_values

class World:
    def __init__(self, cw, ch, population_size, food):
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=cw, height=ch, background='black', scrollregion=(0, 0, cw, ch))

        self.scrollable_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollbar_x = tk.Scrollbar(self.root, orient="horizontal", command=self.canvas.xview)
        self.scrollbar_y = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar_y.pack(side="right", fill="y")
        self.scrollbar_x.pack(side="bottom", fill="x")
        self.root.bind("<Up>", lambda event: self.canvas.yview_scroll(-1, "units"))
        self.root.bind("<Down>", lambda event: self.canvas.yview_scroll(1, "units"))
        self.root.bind("<Left>", lambda event: self.canvas.xview_scroll(-1, "units"))
        self.root.bind("<Right>", lambda event: self.canvas.xview_scroll(1, "units"))

        self.cw = cw
        self.ch = ch

        #Population variables
        self.population_size = population_size
        self.population = [Cell() for i in range(population_size)]
        self.coordinates = [[np.random.randint(cw), np.random.randint(ch)] for i in range(population_size)]
        self.momentums = [[0, 0] for i in range(population_size)]
        self.energies = [cell.maxEnergy for cell in self.population]
        self.angles = [0 for i in range(self.population_size)]
        self.food_per_state = food
        self.food = [[np.random.randint(cw), np.random.randint(ch)] for i in range(food)]

    def create_cell(self, cell):
        if len(self.population) < max_population:
            _cell = Cell(parents=[cell])
            index = self.population.index(cell)
            self.population.append(_cell)
            self.coordinates.append([self.coordinates[index][0]+5, self.coordinates[index][1]+5])
            self.momentums.append([0, 0])
            self.energies.append(_cell.maxEnergy)
            self.angles.append(0)
    
    def draw_cell(self, cell):
        index = self.population.index(cell)
        C = self.coordinates[index]
        self.canvas.create_oval(C[0], C[1], C[0]+cell.size, C[1]+cell.size, fill=cell.color)

    def draw_food(self):
        for x, y in self.food:
            self.canvas.create_oval(x, y, x+5, y+5, fill='light green')
            self.canvas.create_rectangle(x+2, y+2, x+2.5, y+2.5, fill='black')
    
    def update_cell(self, cell):
        index = self.population.index(cell)
        C = self.coordinates[index]
        pI = [C[0], C[1], self.energies[index], self.angles[index]]
        I = [pI[inputs.index(x)] for x in cell.input_meanings]
        output = cell.forward_prop(I)
        if output != []:
            movement = cell.output_meanings[np.argmax(output)]
            if movement == 'F':
                self.coordinates[index][0] -= cell.speed * np.cos(self.angles[index])
                self.coordinates[index][1] -= cell.speed * np.sin(self.angles[index])
            elif movement == 'B':
                self.coordinates[index][0] += cell.speed * np.cos(self.angles[index])
                self.coordinates[index][1] += cell.speed * np.sin(self.angles[index])
            elif movement == 'TL':
                self.angles[index] -= cell.speed
            elif movement == 'TR':
                self.angles[index] += cell.speed
            elif movement == 'S':
                if self.energies[index] >= 80:
                    self.create_cell(cell)
                    self.energies[index] -= 80
        remove_ = []
        for i in range(len(self.food)):
            if np.sqrt(abs(self.coordinates[index][0] - self.food[i][0])**2 + abs(self.coordinates[index][1] - self.food[i][1])**2) < 20:
                self.energies[index] += energy_per_food if self.energies[index] < cell.maxEnergy else 0
                remove_.append(i)
        for i in list(set(remove_)):
            self.food.pop(i)

        if self.coordinates[index][0] > self.cw:
            self.coordinates[index][0] = 2
        elif self.coordinates[index][0] < 0:
            self.coordinates[index][0] = self.cw-cell.size
        elif self.coordinates[index][1] > self.ch:
            self.coordinates[index][1] = 2
        elif self.coordinates[index][1] < 0:
            self.coordinates[index][1] = self.ch-cell.size

    def update(self):
        self.canvas.delete('all')
        self.draw_food()
        for i in self.population:
            self.energies[self.population.index(i)] -= energy_decay_ps/(1000/16)
            self.update_cell(i)
            self.draw_cell(i)
            if self.energies[self.population.index(i)] <= 0:
                index = self.population.index(i)
                self.population.remove(i)
                self.energies.pop(index)
                self.angles.pop(index)
                self.momentums.pop(index)
                self.coordinates.pop(index)
        self.food += [[np.random.randint(self.cw), np.random.randint(self.ch)] for i in range(self.food_per_state-(len(self.food)+1))]

        self.canvas.create_text(65, 10, text=f'Population size: {len(self.population)+1}', fill='white')
        self.root.after(16, self.update)
        
    def start(self):
        self.update()
        self.root.mainloop()

x = World(canvas_width, canvas_height, population_size, food_per_state)
x.start()
