from abc import abstractmethod
import time
import csv
import random

# Base class for all tree nodes
class TreeNode:
    def __init__(self, parent, is_right_child=False):
        self.buffer             = list()  # stores all elements
        self.parent             = parent

        # Evaluation of nodes (their operator) is invoked when an element is present in a nodes right child buffer.
        # That's because events arrive in an ordered manner, and we evaluate sequential patterns.
        # Therefore, when an element is present in a right child buffer, all relevant elements must already be present in the left child buffer.
        # As our evaluation works bottom-up, we let right child nodes signal their parents that they should evaluate.
        # self.is_right_child is used to determine whether this node signals its parent.
        self.is_right_child     = is_right_child

        # nodes store the earliest arrival time, its propagate along with the evaluation signal.
        # This is used to filter events that are out-of-timeframe.
        self.eat                = None

        # these usually can't be assigned during initialization as child nodes are often created after their parents
        self.left_child_buffer  = None
        self.right_child_buffer = None

    # Wire buffers, used when constructing the query plan.
    def set_child_buffers(self, left_buffer, right_buffer):
        self.left_child_buffer  = left_buffer
        self.right_child_buffer = right_buffer

    def get_buffer(self):
        return self.buffer

    # Set and propagate the earliest arrival time.
    # Each node has a right child, thus, only right child need to propagate.
    def set_and_propagate_eat(self, eat):
        self.eat = eat
        if self.is_right_child:
            self.parent.set_and_propagate_eat(self.eat)

    # This method presents the core difference of different TreeNode subclasses.
    # It defines behaviour of a node when it gets signalled to evaluate.
    @abstractmethod
    def eval(self):
        pass


# This node has the purpose to output matching events and collect metrics and results.
# It's similar to the observer design pattern, a call of eval is signaling an assembly round finished
# It does not implement any filtering logic.
# it's the only type that has only one child (the last node that actually implemented some filtering logic)
class RootNode(TreeNode):
    def __init__(self):
        super().__init__(parent=None)

    # Compute and print metrics
    def eval(self):
        # print(f"Number of matches: {len(self.buffer)}")
        pass

    def set_child_buffer(self, buffer):
        self.buffer = buffer

    # This node has nor parent and thus no need to propagate the earliest arrival time.
    def set_and_propagate_eat(self, eat):
        pass


# Node that implements the sequence operator
class SequenceNode(TreeNode):
    def __init__(self, parent, is_right_child=False, condition=False):
        super().__init__(parent, is_right_child)

        # This control whether the hardcoded price predicate should be evaluated
        self.condition = condition

    # Implementation of the SEQ operator
    def eval(self):
        # copy left child buffer as we need to remove objects from it while still iterating over it
        left_buf = self.left_child_buffer.copy()

        for event_right in self.right_child_buffer:
            if event_right['start_timestamp'] < self.eat:  # earliest arrival time filtering
                continue

            for event_left in left_buf:
                if event_left['start_timestamp'] < self.eat:   # earliest arrival time filtering
                    self.left_child_buffer.remove(event_left)  # remove from buffer instead of iterated object (avoids race conditions)
                    continue

                if event_right['start_timestamp'] > event_left['end_timestamp']:  # check sequence property

                    if (not self.condition) or (self.condition and event_left['price'] > event_right['price']):  # check condition (if enabled)
                        # build composite event
                        merged_event = {
                            'start_timestamp': event_left['start_timestamp'],
                            'event'          : event_left['event'] + event_right['event'],
                            'price'          : event_left.get('price', None),  # in our cases prices of the left event are the relevant ones
                            'end_timestamp'  : event_right['end_timestamp']
                        }
                        # append new event to own buffer
                        self.buffer.append(merged_event)
                else:
                    break
            self.left_child_buffer = left_buf  # remove out-of-timeframe events from buffers these will also be out-of-timeframe for future buffers

        # empty right buffer
        self.right_child_buffer.clear()

        # Invoke next layers evaluation if this is a right child node
        if self.is_right_child:
            self.parent.eval()


# Leaf Nodes
class LeafNode(TreeNode):
    def __init__(self, parent, is_right_child=False):
        super().__init__(parent, is_right_child)

    def eval(self):
        if self.is_right_child:
            self.parent.eval()

    def add_2_buffer(self, event):
        self.buffer.append(event)


# query 5 left deep plan construction
# name: prefix of result file
# Returns triple:
#   Root node
#   Dictionary that maps Event types to leaf nodes
#   Final event type that should invoke assembly rounds
def create_q5_ld():

    # init all nodes
    root = RootNode() # this mode is only used as a reference to the final buffer

    seq2 = SequenceNode(parent=root, is_right_child=True)  # right child to invoke root node evaluation 
    seq1 = SequenceNode(parent=seq2)

    leaf_IBM    = LeafNode(parent=seq1)
    leaf_Sun    = LeafNode(parent=seq1, is_right_child=True)
    leaf_Oracle = LeafNode(parent=seq2, is_right_child=True)

    # wire child buffers
    seq1.set_child_buffers(leaf_IBM.get_buffer(), leaf_Sun.get_buffer())
    seq2.set_child_buffers(seq1.get_buffer(), leaf_Oracle.get_buffer())

    root.set_child_buffer(seq2.get_buffer())

    event_type_2_input_map = {"IBM": leaf_IBM, "Sun": leaf_Sun, "Oracle": leaf_Oracle}
    return root, event_type_2_input_map, "Oracle"


# query 5 right deep plan construction
# name: prefix of result file
# Returns triple:
#   Root node
#   Dictionary that maps Event types to leaf nodes
#   Final event type that should invoke assembly rounds
def create_q5_rd():

    # init all nodes
    root = RootNode()  # this mode is only used as a reference to the final buffer

    seq2 = SequenceNode(parent=root, is_right_child=True)  # right child to invoke root node evaluation
    seq1 = SequenceNode(parent=seq2, is_right_child=True)

    leaf_IBM    = LeafNode(parent=seq2)
    leaf_Sun    = LeafNode(parent=seq1)
    leaf_Oracle = LeafNode(parent=seq1, is_right_child=True)

    # wire child buffers
    seq1.set_child_buffers(leaf_Sun.get_buffer(), leaf_Oracle.get_buffer())
    seq2.set_child_buffers(leaf_IBM.get_buffer() , seq1.get_buffer())

    root.set_child_buffer(seq2.get_buffer())

    event_type_2_input_map = {"IBM": leaf_IBM, "Sun": leaf_Sun, "Oracle": leaf_Oracle}
    return root, event_type_2_input_map, "Oracle"


# query 4 left deep plan construction
# name: prefix of result file
# Returns triple:
#   Root node
#   Dictionary that maps Event types to leaf nodes
#   Final event type that should invoke assembly rounds
def create_q4_ld():

    # init all nodes
    root = RootNode()  # this mode is only used as a reference to the final buffer

    seq2 = SequenceNode(parent=root, is_right_child=True)  # right child to invoke root node evaluation
    seq1 = SequenceNode(parent=seq2, condition=True)  # enable condition for query 4

    leaf_IBM    = LeafNode(parent=seq1)
    leaf_Sun    = LeafNode(parent=seq1, is_right_child=True)
    leaf_Oracle = LeafNode(parent=seq2, is_right_child=True)

    # wire child buffers
    seq1.set_child_buffers(leaf_IBM.get_buffer(), leaf_Sun.get_buffer())
    seq2.set_child_buffers(seq1.get_buffer(), leaf_Oracle.get_buffer())

    root.set_child_buffer(seq2.get_buffer())

    event_type_2_input_map = {"IBM": leaf_IBM, "Sun": leaf_Sun, "Oracle": leaf_Oracle}
    return root, event_type_2_input_map, "Oracle"


# query 4 right deep plan construction
# name: prefix of result file
# Returns triple:
#   Root node
#   Dictionary that maps Event types to leaf nodes
#   Final event type that should invoke assembly rounds
def create_q4_rd():

    # init all nodes
    root = RootNode()  # this mode is only used as a reference to the final buffer

    seq2 = SequenceNode(parent=root, is_right_child=True, condition=True)  # right child to invoke root node evaluation
    seq1 = SequenceNode(parent=seq2, is_right_child=True)

    leaf_IBM    = LeafNode(parent=seq2)
    leaf_Sun    = LeafNode(parent=seq1)
    leaf_Oracle = LeafNode(parent=seq1, is_right_child=True)

    # wire child buffers
    seq1.set_child_buffers(leaf_Sun.get_buffer(), leaf_Oracle.get_buffer())
    seq2.set_child_buffers(leaf_IBM.get_buffer() , seq1.get_buffer())

    root.set_child_buffer(seq2.get_buffer())

    event_type_2_input_map = {"IBM": leaf_IBM, "Sun": leaf_Sun, "Oracle": leaf_Oracle}
    return root, event_type_2_input_map, "Oracle"


# function for tree-based evaluation
# window_size specifies time-window
# batch size specifies number of events loaded into leaf buffers until assembly rounds can start
def tree_based_eval(tree_plan, event_stream, window_size=200, batch_size=1):
    root_node, input_map, final_type = tree_plan  # unpack tree plan triple

    # keep track of number of events for batching
    event_count = 0

    # keep track if final event type encountered
    final_type_detected = False

    for e in event_stream:
        event_count += 1

        # get event type and insert into corresponding leaf buffer
        event_type = e['event']
        leaf_node = input_map[event_type]
        leaf_node.add_2_buffer(e)

        # check for final event type
        if event_type == final_type:
            final_type_detected = True

        # check if batch size reached
        if event_count >= batch_size:

            # reset event count
            event_count = 0

            # start assembly round if final event type detected
            if final_type_detected:
                # reset final type detection
                final_type_detected = False

                # compute earliest arrival time
                eat = e['end_timestamp'] - window_size

                # evaluate all leaf nodes (only right child nodes will propagate evaluation)
                for leaf in input_map.values():
                    leaf.set_and_propagate_eat(eat)  # need to propagate eat first
                    leaf.eval()




def gen_input_stream(events, prices, event_rates, length):
    input_stream = []
    timestamp = 1

    for _ in range(length):
        event = random.choices(list(event_rates.keys()), list(event_rates.values()))[0]
        price = random.choice(prices)

        input_stream.append({'start_timestamp': timestamp, 'event': event, 'price': price, 'end_timestamp': timestamp})
        timestamp += 1

    input_stream = sorted(input_stream, key=lambda event: event['start_timestamp'])

    return input_stream

def gen_input_stream_selectivity(events, prices, length, selectivity):
    input_stream = []
    min_price = min(prices)
    max_price = max(prices)

    for i in range(length):
        event = random.choice(events)
        price = random.choice(prices)

        if event == 'Sun':
            price = min_price
            for prev_event in input_stream[i-200:i]:
                if prev_event['event'] == 'IBM' and random.random() < selectivity:
                    prev_event['price'] = max_price

        input_stream.append({'start_timestamp': i, 'event': event, 'price': price, 'end_timestamp': i})

    input_stream = sorted(input_stream, key=lambda event: event['start_timestamp'])

    return input_stream

def rate_experiments(batch_size=1):
    length = 100_000
    events = ["IBM", "Sun", "Oracle"]
    prices = [1, 2, 3]

    # Event Rate Experiment
    rate_list = [{'IBM': 32, 'Sun':   1,  'Oracle':   1},
                 {'IBM': 16, 'Sun':   1,  'Oracle':   1},
                 {'IBM':  1, 'Sun':  16,  'Oracle':  16},
                 {'IBM':  1, 'Sun':  32,  'Oracle':  32},
                 {'IBM':  1, 'Sun':  64,  'Oracle':  64},
                 {'IBM':  1, 'Sun': 128,  'Oracle': 128},
                 {'IBM':  1, 'Sun': 256,  'Oracle': 256}
                 ]
    for _ in range(30):
        for rates in rate_list:
            inpt = gen_input_stream(events, prices, rates, length)
            basename = f"batch_{batch_size}_rate_experiment{rates['IBM']}_{rates['Sun']}_{rates['Oracle']}"

            with open(basename + "q5_rd", 'a', newline='') as f:
                writer = csv.writer(f)
                plan = create_q5_rd()
                t1 = time.time()
                tree_based_eval(plan, inpt, batch_size=batch_size)
                writer.writerow([length/(time.time()-t1)])

            with open(basename + "q5_ld", 'a', newline='') as f:
                writer = csv.writer(f)
                plan = create_q5_ld()
                t1 = time.time()
                tree_based_eval(plan, inpt, batch_size=batch_size)
                writer.writerow([length/(time.time()-t1)])

            with open(basename + "q4_rd", 'a', newline='') as f:
                writer = csv.writer(f)
                plan = create_q4_rd()
                t1 = time.time()
                tree_based_eval(plan, inpt, batch_size=batch_size)
                writer.writerow([length/(time.time()-t1)])

            with open(basename + "q4_ld", 'a', newline='') as f:
                writer = csv.writer(f)
                plan = create_q4_ld()
                t1 = time.time()
                tree_based_eval(plan, inpt, batch_size=batch_size)
                writer.writerow([length/(time.time()-t1)])

def selectivity_experiments(batch_size=1):
    length = 100_000
    events = ["IBM", "Sun", "Oracle"]
    prices = [1, 2, 3]

    # Selectivity Experiment
    selectivities = [1, 1/2, 1/4, 1/8, 1/16, 1/32]

    for _ in range(30):
        for sel in selectivities:
            inpt = gen_input_stream_selectivity(events, prices, length, sel)
            basename = f"batch_{batch_size}_selectivity_experiment{str(round(sel,2)).replace('.', ',')}"

            with open(basename + "q5_rd", 'a', newline='') as f:
                writer = csv.writer(f)
                plan = create_q5_rd()
                t1 = time.time()
                tree_based_eval(plan, inpt, batch_size=batch_size)
                writer.writerow([length / (time.time() - t1)])

            with open(basename + "q5_ld", 'a', newline='') as f:
                writer = csv.writer(f)
                plan = create_q5_ld()
                t1 = time.time()
                tree_based_eval(plan, inpt, batch_size=batch_size)
                writer.writerow([length / (time.time() - t1)])

            with open(basename + "q4_rd", 'a', newline='') as f:
                writer = csv.writer(f)
                plan = create_q4_rd()
                t1 = time.time()
                tree_based_eval(plan, inpt, batch_size=batch_size)
                writer.writerow([length / (time.time() - t1)])

            with open(basename + "q4_ld", 'a', newline='') as f:
                writer = csv.writer(f)
                plan = create_q4_ld()
                t1 = time.time()
                tree_based_eval(plan, inpt, batch_size=batch_size)
                writer.writerow([length / (time.time() - t1)])



if __name__ == '__main__':
    rate_experiments(1)
    selectivity_experiments()

    rate_experiments(100)
    selectivity_experiments(100)

    rate_experiments(1000)
    selectivity_experiments(1000)
