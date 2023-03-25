import pandas       as pd
import tensorflow   as tf
import numpy        as np

from collections.abc                import Callable, Iterable
from tensorflow.python.keras        import Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.metrics.pairwise       import cosine_similarity
from tqdm                           import tqdm


class ATE_Options():
    '''
    Options class for the 
    Approximate Text Explaination (ATE) class, 
    providing configuration details.

    Parameters
    ----------
    column_names: Iterable
        Names of the textual columns that should be used for explainations.
    approximation_steps: int
        Steps for iterative permutations of the base data.
    permutation_border: int
        Total count of permutations per step.
    classes: int
        Number of classes in the classification problem (Binary = 1, Multi-Class/-Label > 2).
    linear_epochs: int
        Count of epochs for the underlying linear model.
    '''

    def __init__(
        self,
        column_names: Iterable,
        approximation_steps: int,
        permutation_border: int,
        classes: int,
        linear_epochs: int
    ) -> None:
        self.column_names = column_names
        self.approximation_steps = approximation_steps
        self.permutation_border = permutation_border
        self.classes = classes
        self.linear_epochs = linear_epochs


class ATE():
    '''
    Approximate Text Explaination (ATE): 
    Use ALTE for local text explaination and 
    AGTE for global text explaination.

    Parameters
    ----------
    tokenize: Callable
        Tokenizer function (str -> np.ndarray[str]).
    vectorize: Callable
        Token-Vectorizer function (str -> np.ndarray[float]).
    classify: Callable
        Classification predict function (pd.DataFrame -> np.ndarray[float]).
    
    Methods
    -------
    transform_effects(Iterable) -> Iterable
        Transforms base model effects into Cosine-Similarity Bag-of-Words metadata.
    '''

    def __init__(
        self, 
        tokenize: Callable,
        vectorize: Callable,
        classify: Callable
    ) -> None:
        self.__tokenize = tokenize
        self.__classify = classify
        self.__vectorize = vectorize

    @staticmethod
    def __apply__(
        series: pd.Series, 
        function: Callable, 
        column_names: Iterable
    ) -> pd.Series:
        for name in column_names:
            series[name] = function(series[name])
        return series
    
    @staticmethod
    def __binary_filter__(
        data: np.ndarray, 
        filter: np.ndarray,
    ) -> np.ndarray:
        return np.array([e for i,e in enumerate(data) if filter[i] == 1])
    
    def __classify__(
        self, 
        permutations: pd.DataFrame
    ) -> Iterable:
        return self.__classify(permutations)

    def __tokenize__(
        self, 
        dataset: pd.DataFrame,
        options: ATE_Options
    ) -> pd.DataFrame:
        return dataset.apply(lambda x: self.__apply__(x, self.__tokenize, options.column_names), axis=1)
    
    @staticmethod
    def __create_linear_model__(inputs: int, options: ATE_Options) -> Model:
        input = Input(shape=(inputs,))
        output = Dense(options.classes, activation='sigmoid')(input)
        model = Model(inputs=input, outputs=output)
        model.compile(optimizer='adam', loss=tf.keras.losses.BinaryFocalCrossentropy(), metrics=['accuracy'])
        return model
    
    @staticmethod
    def __generate_binaries__(
        step: int,
        p_size: int, 
        options: ATE_Options
    ) -> Iterable:
        if step == 0:
            binaries = [list([0]*i+[1]+[0]*(p_size-i-1)) for i in range(p_size)]
            binaries.append(list([1]*p_size))
            binaries.append(list([0]*p_size))
            return [list(e) for e in list(binaries)]
        
        binaries = set()
        while len(binaries) < options.permutation_border:
            binaries.add(tuple(np.random.choice([0, 1], size=(p_size,)).tolist()))
        return [list(e) for e in list(binaries)]
    
    def __create_bow__(self, effects: Iterable) -> dict:
        bow = dict()
        for w, _ in tqdm(effects):
            if not w in bow:
                bow[w] = self.__vectorize(w)
        return bow

    def transform_effects(self, effects: Iterable) -> Iterable:
        '''
        Transforms base model effects into Cosine-Similarity Bag-of-Words metadata.

        Parameters
        ----------
        effects: Iterable
            Base effects from explain function of ALTE or AGTE.

        Returns
        -------
        Iterable
            Cosine-Similarity Bag-of-Words
        '''

        cosine_bow_effects = []
        bow = self.__create_bow__(effects)
        bow_vecs = list(bow.values())
        cosine_bow = cosine_similarity([bow[w] for w,_ in effects], bow_vecs)
        for i, (_, e) in tqdm(enumerate(effects)):
            cosine_bow_effects.append((cosine_bow[i], e))
        return cosine_bow_effects, bow


class ALTE(ATE):
    '''
    Approximate Local Text Explaination (ALTE)

    Parameters
    ----------
    tokenize: Callable
        Tokenizer function (str -> np.ndarray[str]).
    vectorize: Callable
        Token-Vectorizer function (str -> np.ndarray[float]).
    classify: Callable
        Classification predict function (pd.DataFrame -> np.ndarray[float]).
    
    Methods
    -------
    explain(pd.DataFrame, ATE_Options) -> Iterable
        Local explaination for provided datapoint.
    '''

    def __init__(
        self, 
        tokenize: Callable, 
        vectorize: Callable,
        classify: Callable
    ) -> None:
        super().__init__(tokenize, vectorize, classify)
    
    def explain(
        self,
        datapoint: pd.DataFrame,
        options: ATE_Options
    ) -> Iterable:
        '''
        Local explaination for provided datapoint.

        Parameters
        ----------
        datapoint: pd.DataFrame
            Datapoint to be explained.
        options: ATE_Options
            Configuration.
        
        Returns
        -------
        Iterable
            Base effects from explain function of ALTE.
        '''

        tokenized_base = self.__tokenize__(datapoint, options)
        p_sizes = [len(tokenized_base[name][0]) for name in options.column_names]
        p_size = sum(p_sizes, 0)
        linear_model = self.__create_linear_model__(p_size, options)
        for step in tqdm(range(options.approximation_steps)):
            permutations = []
            p_binaries = self.__generate_binaries__(step, p_size, options)
            for binary in p_binaries:
                permutations.append(self.__combine_permutation__(binary, tokenized_base, p_sizes, options))
            permutations = pd.DataFrame(permutations)
            permutations['y'] = self.__classify__(permutations).round()
            linear_model.fit(permutations['binary'].to_list(), permutations['y'].to_list(), epochs=options.linear_epochs)
        return self.__get_effects__(linear_model, tokenized_base, p_sizes, options)
    
    def __combine_permutation__(
            self,
            binary: Iterable,
            tokenized_base: pd.DataFrame, 
            p_sizes: Iterable, 
            options: ATE_Options
        ) -> dict:
        permutation = {'binary': binary}
        acc = 0
        for i, id in enumerate(options.column_names):
            sub_binary = binary[acc:acc + p_sizes[i]]
            permutation[id] = self.__binary_filter__(tokenized_base[id][0], np.array(sub_binary).astype(bool))
            acc = p_sizes[i]
        return permutation
    
    @staticmethod
    def __get_effects__(
        linear_model: Model,
        tokenized_base: pd.DataFrame,
        p_sizes: Iterable, 
        options: ATE_Options
    ) -> Iterable:
        effects = []
        acc = 0
        weights = linear_model.layers[-1].get_weights()[0]
        for i, id in enumerate(options.column_names):
            sequence = tokenized_base[id][0]
            for j, token in enumerate(sequence):
                effects.append((token, weights[acc+j]))
            acc += p_sizes[i]
        return effects


class AGTE(ATE):
    '''
    Approximate Global Text Explaination (AGTE)

    Parameters
    ----------
    tokenize: Callable
        Tokenizer function (str -> np.ndarray[str]).
    vectorize: Callable
        Token-Vectorizer function (str -> np.ndarray[float]).
    classify: Callable
        Classification predict function (pd.DataFrame -> np.ndarray[float]).
    
    Methods
    -------
    explain(pd.DataFrame, ATE_Options) -> Iterable
        Global explaination for provided dataset.
    '''

    def __init__(
        self, 
        tokenize: Callable, 
        vectorize: Callable,
        classify: Callable
    ) -> None:
        super().__init__(tokenize, vectorize, classify)

    def explain(
        self,
        dataset: pd.DataFrame,
        options: ATE_Options
    ) -> Iterable:
        '''
        Global explaination for provided dataset.

        Parameters
        ----------
        dataset: pd.DataFrame
            Dataset to be explained.
        options: ATE_Options
            Configuration.
        
        Returns
        -------
        Iterable
            Base effects from explain function of AGTE.
        '''

        tokenized_base = self.__tokenize__(dataset, options)
        global_effects = []
        for _, row in tqdm(tokenized_base.iterrows()):
            p_sizes = [len(row[name]) for name in options.column_names]
            p_size = sum(p_sizes, 0)
            linear_model = self.__create_linear_model__(p_size, options)
            for step in range(options.approximation_steps):
                permutations = []
                p_binaries = self.__generate_binaries__(step, p_size, options)
                for binary in p_binaries:
                    permutations.append(self.__combine_permutation__(binary, row, p_sizes, options))
                permutations = pd.DataFrame(permutations)
                permutations['y'] = self.__classify__(permutations).round()
                linear_model.fit(permutations['binary'].to_list(), permutations['y'].to_list(), epochs=options.linear_epochs)
            global_effects.extend(self.__get_effects__(linear_model, row, p_sizes, options))
        return global_effects
        
    def __combine_permutation__(
            self,
            binary: Iterable,
            row: pd.Series, 
            p_sizes: Iterable, 
            options: ATE_Options
        ) -> dict:
        permutation = {'binary': binary}
        acc = 0
        for i, id in enumerate(options.column_names):
            sub_binary = binary[acc:acc + p_sizes[i]]
            permutation[id] = self.__binary_filter__(row[id], np.array(sub_binary).astype(bool))
            acc = p_sizes[i]
        return permutation
    
    @staticmethod
    def __get_effects__(
        linear_model: Model,
        row: pd.Series,
        p_sizes: Iterable, 
        options: ATE_Options
    ) -> Iterable:
        effects = []
        acc = 0
        weights = linear_model.layers[-1].get_weights()[0]
        for i, id in enumerate(options.column_names):
            for j, token in enumerate(row[id]):
                effects.append((token, weights[acc+j]))
            acc += p_sizes[i]
        return effects