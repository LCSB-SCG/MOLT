class Matches:
    """
    Class to store the matches between instances in two images of the same channel.
    """

    def __init__(self, background_id, matches=None):
        self.background_id = background_id
        if matches is None:
            # original_to_shared_id is a dictionary with the original instance id as key
            # and some new id shared between multiple original ids as value
            self.original_to_shared_id = {}
            # shared_to_original_id is a dictionary with the shared instance id as key
            # and the original instance id as value, it is the inverse of original_to_shared_id
            self.shared_to_original_id = {}
        else:
            self.original_to_shared_id = matches
            # create an inverse dictionary
            self.shared_to_original_id = {}
            for k, v in matches.items():
                for i in v:
                    if i not in self.shared_to_original_id.keys():
                        self.shared_to_original_id[i] = []
                    self.shared_to_original_id[i].append(k)

    def add_match(self, old_id, new_id):
        if old_id == self.background_id or new_id == self.background_id:
            return
        if old_id not in self.original_to_shared_id.keys():
            self.original_to_shared_id[old_id] = []
        if new_id not in self.shared_to_original_id.keys():
            self.shared_to_original_id[new_id] = []
        self.original_to_shared_id[old_id].append(new_id)
        self.shared_to_original_id[new_id].append(old_id)


class CrossChannelMatches:
    def __init__(self, channel_a_id=None, channel_b_id=None):
        """
        Save the instance matches between two channels in bi-directional format.

        Args:
            channel_a_id (str): The id of the first channel.
            channel_b_id (str): The id of the second channel.
        """
        self.channel_a_id = channel_a_id
        self.channel_b_id = channel_b_id
        self.a_to_b = {}  # channel_a_id: set[#channel_b_ids]
        self.b_to_a = {}  # channel_b_id: set[#channel_a_ids]

    def add_match(self, channel_a_instance_id=None, channel_b_instance_id=None):
        # at least one of the channel ids must be provided
        if channel_a_instance_id is None and channel_b_instance_id is None:
            raise ValueError("At least one channel id must be provided.")

        if channel_a_instance_id is not None:
            if channel_a_instance_id not in self.a_to_b.keys():
                # initate the channel_a_to_b mapping if it does not exist yet
                self.a_to_b[channel_a_instance_id] = set()

        if channel_b_instance_id is not None:
            if channel_b_instance_id not in self.b_to_a.keys():
                # initate the channel_b_to_a mapping if it does not exist yet
                self.b_to_a[channel_b_instance_id] = set()

        if channel_a_instance_id is not None and channel_b_instance_id is not None:
            # both channel ids are provided, so there is a match between the two channels
            self.a_to_b[channel_a_instance_id].add(channel_b_instance_id)
            self.b_to_a[channel_b_instance_id].add(channel_a_instance_id)
