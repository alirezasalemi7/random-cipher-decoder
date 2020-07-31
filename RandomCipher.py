from numpy.random import shuffle,choice
from numpy import vectorize,where
from string import ascii_lowercase,punctuation,ascii_letters
from re import split
import functools

class Decoder(object):
    class Vectorize(vectorize):
        def __get__(self, obj, objtype):
            return functools.partial(self.__call__, obj)

    def __init__(self,data):
        self.encoded = data
        self.encoded_seprated = self.seprate_coded_data()
        self.pop_size = 100
        self.full_fit = False
        self.mutation_rate = 1/self.pop_size
        self.crossover_rate = 0.9
        self.word_dict = self.make_word_dict()
        self.population = self.make_first_generation()
        self.generation_count = 0
        self.cur_max = 0
        self.total_max = 0
        self.same_max_generation_counter = 0
        self.tops = int(0.3*self.pop_size)

    def make_first_generation(self):
        pop_list = []
        ascii_list = list(ascii_lowercase)
        for i in range(self.pop_size):
            shuffle(ascii_list)
            pop_list.append("".join(ascii_list))
        return pop_list

    def seprate_coded_data(self):
        seprated = split('[\' \n\t\r]+',self.encoded.translate(str.maketrans(punctuation,' '*len(punctuation))))
        _set = {s.lower() for s in seprated if s.isalpha()}
        return _set

    @Vectorize
    def fitness_function(self,genome):
        count = 0
        length = 0
        for i in self.encoded_seprated:
            translated = i.translate(str.maketrans(ascii_lowercase,genome,''))
            if translated in self.word_dict:
                length += len(translated)
                count+=1
        if count == len(self.encoded_seprated):
            self.full_fit = True
        if length > self.cur_max:
            self.cur_max = length
        return length**2*count

    def select_parents(self):
        fits = self.fitness_function(self.population)
        self.population = [x for _,x in sorted(zip(fits.tolist(),self.population))]
        fits=sorted(fits)[self.pop_size-self.tops:]
        fits_sum = sum(fits)
        if fits_sum:
            fits = fits/fits_sum
        choosen = choice(self.population[self.pop_size-self.tops:],size=self.pop_size,p=fits)
        return choosen.tolist();

    def make_next_generation(self):
        next_genertion = []
        ranks = self.select_parents()
        while len(ranks):
            gen1 , gen2 = ranks.pop() , ranks.pop()
            new_gen1 , new_gen2 = self.cross_over(gen1,gen2)
            mutated_new_gen1 , mutated_new_gen2 = self.mutate(new_gen1) , self.mutate(new_gen2)
            next_genertion.append(mutated_new_gen1)
            next_genertion.append(mutated_new_gen2)
        self.generation_count+=1
        return next_genertion

    def cross_over(self,genome1,genome2):
        keep_parent = choice([True,False],p=[1-self.crossover_rate,self.crossover_rate])
        if not keep_parent and not self.full_fit:
            sepration_point = choice(range(1,len(genome1)-1))
            new_genome1 = list(genome1[:sepration_point])+list(genome2[sepration_point:])
            new_genome2 = list(genome2[:sepration_point])+list(genome1[sepration_point:])
            not_seen1 = set(genome1[sepration_point:])
            not_seen2 = set(genome2[sepration_point:])
            seen1 = set(genome1[:sepration_point])
            seen2 = set(genome2[:sepration_point])
            for i in range(sepration_point,len(genome1)):
                if new_genome1[i] in seen1:
                    new_genome1[i] = not_seen1.pop()
                    seen1.add(new_genome1[i])
                else:
                    not_seen1.remove(new_genome1[i])
                    seen1.add(new_genome1[i])
            for i in range(sepration_point,len(genome2)):
                if new_genome2[i] in seen2:
                    new_genome2[i] = not_seen2.pop()
                    seen2.add(new_genome2[i])
                else:
                    not_seen2.remove(new_genome2[i])
                    seen2.add(new_genome2[i])
            return "".join(new_genome1),"".join(new_genome2)
        else:
            return genome1,genome2

    def mutate(self,genome):
        is_mutate = choice([False,True],p=[1-self.mutation_rate,self.mutation_rate])
        if is_mutate and not self.full_fit:
            x,y = choice(list(ascii_lowercase),size=2,replace=False) 
            return genome.translate(str.maketrans(x+y,y+x,''))
        else:
            return genome
    
    def make_word_dict(self):
        with open('global_text.txt') as file:
            content = file.read()
            puncs = punctuation.replace('\'','')
            dic_word = split('[\' \n\t\r]+',content.translate(str.maketrans(puncs,' '*len(puncs))))
            dic_set = set()
            for data in dic_word:
                if len(data)!=0:
                    dic_set.add(data.lower())
            return dic_set         

    def get_best_genome(self):
        fits = self.fitness_function(self.population)
        return self.population[fits.tolist().index(max(fits))]


    def decode(self):
        while not self.full_fit:
            self.population = self.make_next_generation()
            # print(self.generation_count,self.cur_max,self.mutation_rate,self.get_best_genome())
            if self.cur_max <= self.total_max:
                self.mutation_rate = 0.5
                self.same_max_generation_counter += 1
            else:
                self.total_max = self.cur_max
                self.mutation_rate = 1/self.pop_size
                self.same_max_generation_counter = 0
            if self.same_max_generation_counter > 100:
                self.population = self.make_first_generation()
                self.same_max_generation_counter = 0
                self.total_max = 0
            self.cur_max = 0
        best_genome = self.get_best_genome()
        mapping = best_genome + best_genome.upper()
        return self.encoded.translate(str.maketrans(ascii_letters,mapping,''))