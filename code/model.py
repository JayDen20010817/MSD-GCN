from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import torch_geometric
from torch_geometric.nn import HeteroConv
from gensim.models import KeyedVectors
import torchtext.vocab as vocab
class MultimodalEncoder(nn.Module):
    def __init__(self, config, layer_number):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_number)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_encoder_attentions = []
        for layer_module in self.layer:
            hidden_states, attention = layer_module(hidden_states, attention_mask, output_attentions=True)
            all_encoder_attentions.append(attention)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_attentions


#-------------------CBAM------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        self.avg_pool = nn.AdaptiveAvgPool2d(1).to(device)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio).to(device),
            nn.ReLU(inplace=True).to(device),
            nn.Linear(in_channels // reduction_ratio, in_channels).to(device)
        )
        self.sigmoid = nn.Sigmoid().to(device)

    def forward(self, x):
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c).to(device)
        y = self.fc(y).view(b, c, 1, 1).to(device)
        return x * self.sigmoid(y).to(device)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3).to(device)
        self.sigmoid = nn.Sigmoid().to(device)

    def forward(self, x):
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        max_pool = torch.max(x, dim=1, keepdim=True)[0].to(device)
        avg_pool = torch.mean(x, dim=1, keepdim=True).to(device)
        y = torch.cat([max_pool, avg_pool], dim=1).to(device)
        y = self.conv(y).to(device)
        return x * self.sigmoid(y).to(device)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
#--------------------------------------------------------------    

class MSD_MODEL(nn.Module):
    def __init__(self, args):
        super(MSD_MODEL, self).__init__()
        self.model = RoBertaModel.from_pretrained("")
        self.config = BertConfig.from_pretrained("")
        self.config.hidden_size = 512
        self.config.num_attention_heads = 8
        self.trans = MultimodalEncoder(self.config, layer_number=args.layers)
        self.conv=HeteroConv({
            'text': MyConv(512, 512),
            'image': MyConv(512, 512)
        }) # type: ignore
        if args.simple_linear:
            self.text_linear =  nn.Linear(args.text_size, args.text_size)
            self.image_linear =  nn.Linear(args.image_size, args.image_size)
        else:
            self.text_linear =  nn.Sequential(
                nn.Linear(args.text_size, args.text_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )
            self.image_linear =  nn.Sequential(
                nn.Linear(args.image_size, args.image_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )


        self.classifier_fuse = nn.Linear(args.text_size , args.label_number)
        self.classifier_text = nn.Linear(args.text_size, args.label_number)
        self.classifier_image = nn.Linear(args.image_size, args.label_number)

        self.loss_fct = nn.CrossEntropyLoss()
        self.att = nn.Linear(args.text_size, 1, bias=False)
        
    def forward(self, inputs, labels):
        output = self.model(**inputs,output_attentions=True) # type: ignore

        text_features = output['text_model_output']['last_hidden_state']

        image_features = output['vision_model_output']['last_hidden_state']

        text_feature = output['text_model_output']['pooler_output']

        image_feature = output['vision_model_output']['pooler_output']

        text_feature = self.text_linear(text_feature)

        image_feature = self.image_linear(image_feature)

        text_embeds = self.model.text_projection(text_features) 

        image_embeds = self.model.visual_projection(image_features) 

        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        input_embeds = torch.cat((image_embeds, text_embeds), dim=1)


        input_embeds = torch.unsqueeze(input_embeds, dim=2).to(device)
        input_embeds=input_embeds.expand(-1, -1, 512, -1).permute(0, 2, 3, 1).to(device)
        cbam_model = CBAM(in_channels=512)
        input_embeds = cbam_model(input_embeds).permute(0, 3, 2, 1)[:,:,:,0:1].squeeze(3).to(device)



        attention_mask = torch.cat((torch.ones(text_features.shape[0], 50).to(text_features.device), inputs['attention_mask']), dim=-1)
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        fuse_hiddens, all_attentions = self.trans(input_embeds, extended_attention_mask, output_all_encoded_layers=False)
        fuse_hiddens = fuse_hiddens[-1]
        new_text_features = fuse_hiddens[:, 50:, :]

        new_text_feature = new_text_features[
            torch.arange(new_text_features.shape[0], device=inputs['input_ids'].device), inputs['input_ids'].to(torch.int).argmax(dim=-1)
        ]

        new_image_feature = fuse_hiddens[:, 0, :].squeeze(1)

        text_weight = self.att(new_text_feature)
        image_weight = self.att(new_image_feature)

        att = nn.functional.softmax(torch.stack((text_weight, image_weight), dim=-1),dim=-1)
        tw, iw = att.split([1,1], dim=-1)

        fuse_feature = tw.squeeze(1) * new_text_feature + iw.squeeze(1) * new_image_feature
        
        logits_fuse = self.classifier_fuse(fuse_feature)
        logits_text = self.classifier_text(text_feature)
        logits_image = self.classifier_image(image_feature)
   
        fuse_score = nn.functional.softmax(logits_fuse, dim=-1)
        text_score = nn.functional.softmax(logits_text, dim=-1)
        image_score = nn.functional.softmax(logits_image, dim=-1)

        score = fuse_score + text_score + image_score

        outputs = (score,)
        if labels is not None:
            loss_fuse = self.loss_fct(logits_fuse, labels)
            loss_text = self.loss_fct(logits_text, labels)
            loss_image = self.loss_fct(logits_image, labels)
            loss = loss_fuse + loss_text + loss_image

            outputs = (loss,) + outputs
        return outputs


