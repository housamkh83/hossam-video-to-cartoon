import torch
import torch.nn as nn
import torch.nn.functional as F
# (قد لا تحتاج لـ os و Path هنا إذا كان هذا الملف منفصلاً)
# import os
# from pathlib import Path

# --- هذا هو تعريف المولد (Generator) الصحيح ---
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # Encoder layers with reflection padding and explicit naming
        self.refpad01_1 = nn.ReflectionPad2d(3)
        self.conv01_1 = nn.Conv2d(3, 64, 7)
        self.in01_1 = InstanceNormalization(64) # Uses the custom InstanceNormalization below
        # relu will be applied in forward pass

        self.conv02_1 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv02_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.in02_1 = InstanceNormalization(128)
        # relu

        self.conv03_1 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv03_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.in03_1 = InstanceNormalization(256)
        # relu

        # Residual blocks (defined with explicit padding and conv layers)
        # Block 1
        self.refpad04_1 = nn.ReflectionPad2d(1)
        self.conv04_1 = nn.Conv2d(256, 256, 3)
        self.in04_1 = InstanceNormalization(256)
        self.refpad04_2 = nn.ReflectionPad2d(1)
        self.conv04_2 = nn.Conv2d(256, 256, 3)
        self.in04_2 = InstanceNormalization(256)

        # Block 2
        self.refpad05_1 = nn.ReflectionPad2d(1)
        self.conv05_1 = nn.Conv2d(256, 256, 3)
        self.in05_1 = InstanceNormalization(256)
        self.refpad05_2 = nn.ReflectionPad2d(1)
        self.conv05_2 = nn.Conv2d(256, 256, 3)
        self.in05_2 = InstanceNormalization(256)

        # Block 3
        self.refpad06_1 = nn.ReflectionPad2d(1)
        self.conv06_1 = nn.Conv2d(256, 256, 3)
        self.in06_1 = InstanceNormalization(256)
        self.refpad06_2 = nn.ReflectionPad2d(1)
        self.conv06_2 = nn.Conv2d(256, 256, 3)
        self.in06_2 = InstanceNormalization(256)

        # Block 4
        self.refpad07_1 = nn.ReflectionPad2d(1)
        self.conv07_1 = nn.Conv2d(256, 256, 3)
        self.in07_1 = InstanceNormalization(256)
        self.refpad07_2 = nn.ReflectionPad2d(1)
        self.conv07_2 = nn.Conv2d(256, 256, 3)
        self.in07_2 = InstanceNormalization(256)

        # Block 5
        self.refpad08_1 = nn.ReflectionPad2d(1)
        self.conv08_1 = nn.Conv2d(256, 256, 3)
        self.in08_1 = InstanceNormalization(256)
        self.refpad08_2 = nn.ReflectionPad2d(1)
        self.conv08_2 = nn.Conv2d(256, 256, 3)
        self.in08_2 = InstanceNormalization(256)

        # Block 6
        self.refpad09_1 = nn.ReflectionPad2d(1)
        self.conv09_1 = nn.Conv2d(256, 256, 3)
        self.in09_1 = InstanceNormalization(256)
        self.refpad09_2 = nn.ReflectionPad2d(1)
        self.conv09_2 = nn.Conv2d(256, 256, 3)
        self.in09_2 = InstanceNormalization(256)

        # Block 7
        self.refpad10_1 = nn.ReflectionPad2d(1)
        self.conv10_1 = nn.Conv2d(256, 256, 3)
        self.in10_1 = InstanceNormalization(256)
        self.refpad10_2 = nn.ReflectionPad2d(1)
        self.conv10_2 = nn.Conv2d(256, 256, 3)
        self.in10_2 = InstanceNormalization(256)

        # Block 8
        self.refpad11_1 = nn.ReflectionPad2d(1)
        self.conv11_1 = nn.Conv2d(256, 256, 3)
        self.in11_1 = InstanceNormalization(256)
        self.refpad11_2 = nn.ReflectionPad2d(1)
        self.conv11_2 = nn.Conv2d(256, 256, 3)
        self.in11_2 = InstanceNormalization(256)

        # Decoder layers
        self.deconv01_1 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv01_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.in12_1 = InstanceNormalization(128) # Note the InstanceNorm name index

        self.deconv02_1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv02_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.in13_1 = InstanceNormalization(64) # Note the InstanceNorm name index

        self.refpad12_1 = nn.ReflectionPad2d(3)
        self.deconv03_1 = nn.Conv2d(64, 3, 7) # Final conv layer

    def forward(self, x):
        # Encoder
        y = F.relu(self.in01_1(self.conv01_1(self.refpad01_1(x))))
        y = F.relu(self.in02_1(self.conv02_2(self.conv02_1(y))))
        t04 = F.relu(self.in03_1(self.conv03_2(self.conv03_1(y)))) # Output of encoder

        # Residual blocks application
        y = F.relu(self.in04_1(self.conv04_1(self.refpad04_1(t04))))
        t05 = self.in04_2(self.conv04_2(self.refpad04_2(y))) + t04

        y = F.relu(self.in05_1(self.conv05_1(self.refpad05_1(t05))))
        t06 = self.in05_2(self.conv05_2(self.refpad05_2(y))) + t05

        y = F.relu(self.in06_1(self.conv06_1(self.refpad06_1(t06))))
        t07 = self.in06_2(self.conv06_2(self.refpad06_2(y))) + t06

        y = F.relu(self.in07_1(self.conv07_1(self.refpad07_1(t07))))
        t08 = self.in07_2(self.conv07_2(self.refpad07_2(y))) + t07

        y = F.relu(self.in08_1(self.conv08_1(self.refpad08_1(t08))))
        t09 = self.in08_2(self.conv08_2(self.refpad08_2(y))) + t08

        y = F.relu(self.in09_1(self.conv09_1(self.refpad09_1(t09))))
        t10 = self.in09_2(self.conv09_2(self.refpad09_2(y))) + t09

        y = F.relu(self.in10_1(self.conv10_1(self.refpad10_1(t10))))
        t11 = self.in10_2(self.conv10_2(self.refpad10_2(y))) + t10

        y = F.relu(self.in11_1(self.conv11_1(self.refpad11_1(t11))))
        y = self.in11_2(self.conv11_2(self.refpad11_2(y))) + t11 # Final output of res blocks

        # Decoder
        y = F.relu(self.in12_1(self.deconv01_2(self.deconv01_1(y))))
        y = F.relu(self.in13_1(self.deconv02_2(self.deconv02_1(y))))
        y = F.tanh(self.deconv03_1(self.refpad12_1(y))) # Final output (Tanh activation)

        return y

# --- هذا هو تعريف Instance Normalization المخصص المستخدم في الموديل ---
class InstanceNormalization(nn.Module):
    """Instance Normalization class (redefined for clarity and potential minor differences)"""
    def __init__(self, dim, eps=1e-9):
        super(InstanceNormalization, self).__init__()
        # These are learnable parameters specific to Instance Norm
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize scale to uniform random and shift to zero."""
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def forward(self, x):
        """Apply instance normalization."""
        # Calculate mean and variance across spatial dimensions (height and width)
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        # Calculate the biased variance (important distinction from torch.var's default)
        var = torch.var(t, 2, unbiased=False).unsqueeze(2).unsqueeze(3).expand_as(x) # Use unbiased=False for biased var

        # Prepare scale and shift parameters for broadcasting
        scale_broadcast = self.scale.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        shift_broadcast = self.shift.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)

        # Apply normalization formula
        out = (x - mean) / torch.sqrt(var + self.eps)
        # Apply scale and shift
        out = out * scale_broadcast + shift_broadcast
        return out

# --- هذا الكلاس غير مستخدم مباشرة من قبلنا، لكنه موجود في الملف الأصلي ---
# --- نحن نستخدم Transformer مباشرة ---
# class Model():
#     def __init__(self, model_name, device) -> None:
#         self._device = device
#         self._model = Transformer()
#         path = os.path.join(str(Path(__file__).parent), 'weights', model_name + '_net_G_float.pth')
#         self._model.load_state_dict(torch.load(path))
#         self._model.to(self._device)
#         self._model.eval()

#     def __call__(self, img_tensor: torch.Tensor): # Added type hint
#         img_tensor = img_tensor.to(self._device)
#         # Normalize input tensor from [0, 1] to [-1, 1]
#         img_tensor = img_tensor * 2.0 - 1.0 # Correct normalization

#         output_image = self._model(img_tensor)
#         output_image = output_image[0] # Get the first image from the batch

#         # Assuming the model outputs RGB, no need to swap channels here if input was RGB
#         # If the model was trained expecting BGR, then swap might be needed depending on input format
#         # output_image = output_image[[2, 1, 0], :, :] # Keep commented unless input is BGR

#         # Denormalize output from [-1, 1] to [0, 1]
#         output_image = output_image.data.cpu().float() * 0.5 + 0.5
#         return output_image.numpy()

# --- قم بتغيير اسم الكلاس الرئيسي هنا ليطابق ما يتوقعه كود الواجهة ---
# --- الكود الأصلي يستخدم Transformer, لكن كود الواجهة يتوقع Generator ---
# --- أسهل حل هو إعادة تسمية Transformer إلى Generator ---
class Generator(Transformer): # Inherit from Transformer, effectively renaming it
    def __init__(self):
        super(Generator, self).__init__() # Call Transformer's __init__