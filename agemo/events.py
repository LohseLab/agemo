import collections
import itertools


class Event:
    """
    Super class for all events. Note that all events require an index.
    This unique integer tells both `agemo` and the user at which position
    in the variable and equation array the coefficients for this event
    will be stored.

    :param idx: Index of variable in variable and equation array.
    :type idx: int
    :param discrete: indicates whether an event is discrete or continuous.
    :type discrete: boolean

    """

    def __init__(self, idx, discrete, **kwargs):
        self.idx = idx
        self._discrete = discrete
        self.__dict__.update(kwargs)

    def _single_step(self):
        raise NotImplementedError

    @property
    def discrete(self):
        return self._discrete


class MigrationEvent(Event):
    """
    Represents unidirectional migration between source and destination
    for the coalescent process. Note that migration is specified backwards
    in time. Effect: at each step in the state space a single lineage can
    move from source to destination.

    :param idx: Index of variable in variable and equation array.
    :type idx: int
    :param source: Integer representing the index of the population,
        as defined in `sample_configuration` acting as source.
    :type source: int
    :param int destination: Integer representing the index of the population,
        as defined in `sample_configuration` acting as destination.
    :type destination: int

    """

    def __init__(self, idx, source, destination):
        super().__init__(idx, discrete=False, source=source, destination=destination)

    def _single_step(self, state_list):
        result = []
        lineage_count = collections.Counter(state_list[self.source])
        for lineage, count in lineage_count.items():
            temp = list(state_list)
            idx = temp[self.source].index(lineage)
            temp[self.source] = tuple(
                temp[self.source][:idx] + temp[self.source][idx + 1 :]
            )
            temp[self.destination] = tuple(
                sorted(
                    list(temp[self.destination])
                    + [
                        lineage,
                    ]
                )
            )
            result.append((self.idx, count, tuple(temp)))
        return result


class PopulationSplitEvent(Event):
    """
    Represents a population split from an ancestral population
    into two or more derived populations. Effect: all lineages are
    moved from the derived population(s) into the ancestral population
    backwards in time.

    :param idx: Index of variable in variable and equation array.
    :type idx: int
    :param int ancestral: Integer of population, as specified in `sample_configuration`,
        representing the ancestral population.
    :type ancestral: int
    :param derived: One or more integers representing
        index/indices of derived populations as specified in `sample_configuration`.
    :type derived: int

    """

    def __init__(self, idx, ancestral, *derived):
        super().__init__(idx, discrete=True, derived=derived, ancestral=ancestral)

    def _single_step(self, state_list):
        result = []
        temp = list(state_list)
        sources_joined = tuple(
            itertools.chain.from_iterable([state_list[idx] for idx in self.derived])
        )
        if len(sources_joined) > 0:
            temp[self.ancestral] = tuple(
                sorted(state_list[self.ancestral] + sources_joined)
            )
            for idx in self.derived:
                temp[idx] = ()
            result.append((self.idx, 1, tuple(temp)))
        return result


class CoalescenceEvent(Event):
    """
    Describing all coalescence events among the lineages as defined in
    `sample_configuration`.

    :param idx: Index of variable in variable and equation array.
    :type idx: int

    """

    def __init__(self, idx):
        super().__init__(idx, discrete=False)

    def coalesce_lineages(self, input_tuple, to_join):
        """
        Joins lineages and returns resulting tuple of lineages for a single pop

        """
        result = list(input_tuple)
        for lineage in to_join:
            result.remove(lineage)
        result.append("".join(sorted(flatten(to_join))))
        result.sort()
        return tuple(result)

    def _single_step(self, pop_state):
        """
        For single population generate all possible coalescence events
        param: iterable pop_state: containing all lineages (str) present within pop
        param: float, object coal_rate: rate at which coalescence happens in that pop

        """
        coal_event_pairs = list(itertools.combinations(pop_state, 2))
        coal_counts = collections.Counter(coal_event_pairs)
        for lineages, count in coal_counts.items():
            result = self.coalesce_lineages(pop_state, lineages)
            yield (count, result)


class EventsSuite:
    def __init__(self, events):
        self.events = events

    def __len__(self):
        return len(self.events)

    def _single_step(self, state_list):
        raise NotImplementedError


class CoalescenceEventsSuite(EventsSuite):
    """
    Class groups all coalescence events for a particular model.

    :param num_coalescence_events: Number of populations in your `sample_configuration`
    :type num_coalescence_events: int
    :param idxs: List of indices for each of the coalescence events, defaults to None.
        Should be of length `num_coalescence_events`
    :type idxs: list(int)

    """

    def __init__(self, num_coalescence_events, idxs=None):
        if idxs is None:
            coalescence_events = [
                CoalescenceEvent(i) for i in range(num_coalescence_events)
            ]
        else:
            coalescence_events = [CoalescenceEvent(i) for i in idxs]
        super().__init__(coalescence_events)

    def _single_step(self, state_list):
        result = []
        for idx, (pop, coalescence_event) in enumerate(zip(state_list, self.events)):
            for count, single_event in coalescence_event._single_step(pop):
                modified_state_list = list(state_list)
                modified_state_list[idx] = single_event
                result.append(
                    (coalescence_event.idx, count, tuple(modified_state_list))
                )
        return result


def flatten(input_list):
    """
    Flattens iterable from depth n to depth n-1
    :param list input_list: iterable that will be flattened.

    """
    return itertools.chain.from_iterable(input_list)
