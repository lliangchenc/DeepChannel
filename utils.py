import torch

def recursive_to_device(device, *tensors):
    return [recursive_to_device(device, *t) if isinstance(t, list) or isinstance(t, tuple) \
            else t.to(device) for t in tensors]

def visualize_tensor(tensor):
    tensor = tensor.squeeze()
    if tensor.dim() == 0:
        return '%.3f' % tensor.item()
    elif tensor.dim() == 1:
        return ' '.join(['%.3f' % x for x in tensor])
    elif tensor.dim() == 2:
        return '\n'.join([visualize_tensor(tensor[i]) for i in range(tensor.size(0))])
    else:
        raise Exception('dim must <=2')

def reverse_padded_sequence(inputs, lengths, batch_first=False):
    """Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Arguments:
        inputs (Variable): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A Variable with the same size as inputs, but with each sequence
        reversed according to its length.
    """

    if not batch_first:
        inputs = inputs.transpose(0, 1)
    if inputs.size(0) != len(lengths):
        raise ValueError('inputs incompatible with lengths.')
    reversed_indices = [list(range(inputs.size(1)))
                        for _ in range(inputs.size(0))]
    for i, length in enumerate(lengths):
        if length > 0:
            reversed_indices[i][:length] = reversed_indices[i][length-1::-1]
    reversed_indices = (torch.LongTensor(reversed_indices).unsqueeze(2)
                        .expand_as(inputs))
    reversed_indices = reversed_indices.to(inputs.device)
    reversed_inputs = torch.gather(inputs, 1, reversed_indices)
    if not batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs
