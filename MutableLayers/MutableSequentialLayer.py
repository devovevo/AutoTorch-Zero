import torch

from MutableLayer import Mutable

from CompoundLayers.SequentialLayer import Sequential

class MutableSequential(Sequential, Mutable):
    def apply_to_layer(self, action, index, new_layer=None):
        cur_index = 0

        for i, layer in enumerate(self.layers):
            if cur_index == index:
                match action:
                    case "insert":
                        self.layers.insert(i, new_layer)
                    case "remove":
                        self.layers.pop(i)
                    case "replace":
                        self.layers[i] = new_layer
                break
            elif cur_index < index and cur_index + layer.num_layers() > index and isinstance(layer, Mutable):
                index_within = index - cur_index - 1

                match action:
                    case "insert":
                        layer.insert_layer(index_within, new_layer)
                    case "remove":
                        layer.remove_layer(index_within)
                    case "replace":
                        layer.replace_layer(index_within, new_layer)
                break

            cur_index += layer.num_layers()

    def insert_layer(self, index, layer):
        self.apply_to_layer('insert', index, layer)

    def remove_layer(self, index):
        if self.num_layers() > 1:
            self.apply_to_layer('remove', index)
    
    def replace_layer(self, index, layer):
        self.apply_to_layer('replace', index, layer)

    def copy(self):
        return MutableSequential(self.dim, [layer.copy() for layer in self.layers])