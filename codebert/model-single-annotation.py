# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch

class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, encoder,decoder,config,beam_size=None,max_length=None,sos_id=None,eos_id=None,dot_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()
        
        self.beam_size=beam_size
        self.max_length=max_length
        self.sos_id=sos_id
        self.eos_id=eos_id
        
    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
                  
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)        
        
    def forward(self, source_ids=None,source_mask=None,target_ids=None,target_mask=None,args=None,
                     context_api_ids=None, context_api_mask=None):   

        outputs = self.encoder(source_ids, attention_mask=source_mask) # B * L * D
        encoder_output = outputs[0].permute([1,0,2]).contiguous() # L * B * D
        if target_ids is not None:  
            attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(target_ids).permute([1,0,2]).contiguous() # L * B * D
            out = self.decoder(tgt_embeddings,encoder_output,tgt_mask=attn_mask,memory_key_padding_mask=(1-source_mask).bool()) # L * B * D
            hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous() # B * L * D
            lm_logits = self.lm_head(hidden_states) # B * L * V
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1 # [B*(L-1)]
            shift_logits = lm_logits[..., :-1, :].contiguous() # B * L-1 * V
            shift_labels = target_ids[..., 1:].contiguous() # B * L-1
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss]) # [B*(L-1)] * V , [B*(L-1)]

            outputs = loss,loss*active_loss.sum(),active_loss.sum()
            return outputs
        else:
            #Predict 
            preds=[]       
            zero=torch.cuda.LongTensor(1).fill_(0)     # [0]
            for i in range(source_ids.shape[0]): # B
                # if i > 3:
                #     continue

                context=encoder_output[:,i:i+1] # L * 1 * D
                context_mask=source_mask[i:i+1,:] # 1 * L
                context_api = []
                context_api_count = 0
                has_context = False

                if len(context_api_ids) > 0:
                    context_api = context_api_ids[i:i + 1, :][0].tolist()
                    
                    context_api_id_mask = context_api_mask[i:i+1, :][0].tolist()

                    if 0 in context_api_id_mask:
                        context_api = context_api[:context_api_id_mask.index(0)]
                    else:
                        print(f"### target full of context_api, context_api: {context_api}")

                    context_api.pop(0)
                    context_api_count = len(context_api)
                    if context_api_count > 0:
                        has_context = True

                beam = Beam(self.beam_size, self.sos_id, self.eos_id, context_api_len=context_api_count)
                input_ids=beam.getCurrentState() # beam * 1
                context=context.repeat(1, self.beam_size,1) # L * beam * D
                context_mask=context_mask.repeat(self.beam_size,1) # beam * L

                # is_context_exist = False
                # print(f"context_api: {context_api}")
                for _ in range(len(context_api) + self.max_length): 
                # for _ in range(self.max_length): 
                    if beam.done():
                        # print(f"beam break")
                        break
                    attn_mask=-1e4 *(1-self.bias[:input_ids.shape[1],:input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(input_ids).permute([1,0,2]).contiguous() # 1 * beam * D
                    out = self.decoder(tgt_embeddings,context,tgt_mask=attn_mask,memory_key_padding_mask=(1-context_mask).bool()) # 1 * beam * D
                    out = torch.tanh(self.dense(out))
                    hidden_states=out.permute([1,0,2]).contiguous()[:,-1,:] # beam * D
                    out = self.lsm(self.lm_head(hidden_states)).data # beam * V

                    if context_api:
                        # is_context_exist = True
                        context_api_id = context_api.pop(0)
                        beam.advance(out, context_api_id)
                        # beam.advance(out)
                    else:
                        # if is_context_exist:
                        #     beam.advance(out)
                        # else:
                        #     is_context_exist = False
                        if has_context:
                            beam.advance(out, generate_beam=False)
                            has_context = False
                        else:
                            beam.advance(out)

                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin())) # beam * 1
                    input_ids=torch.cat((input_ids,beam.getCurrentState()),-1) # beam * 2

                hyp= beam.getHyp(beam.getFinal())
                # print(f"hyp: {hyp}")
                pred=beam.buildTargetTokens(hyp)[:self.beam_size]

                # print(f"pred: {pred}")
                pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred] # beam * l
                preds.append(torch.cat(pred,0).unsqueeze(0)) # 1 * beam * l
                
            preds=torch.cat(preds,0)                # B * beam * l
            # print(f"preds: {preds}")
            return preds
        
        

class Beam(object):
    def __init__(self, size,sos,eos,context_api_len=0):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []
        self.context_api_len = context_api_len

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk, context_api=None, generate_beam=True):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)
        is_search = True

        # Sum the previous scores.
        tempBeamLk = wordLk[0]

        if context_api:
            last_context_api = True

        if len(self.prevKs) > 0:
            if context_api or not generate_beam:
                beamLk = wordLk[0]
                beamLk[self._eos] = -1e20 #prevent eos
                # beamLk[context_api] = 1
            else:
                beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)                    

                # Don't let EOS have children.
                for i in range(self.nextYs[-1].size(0)):
                    if self.nextYs[-1][i] == self._eos:
                        beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
            idx = 0

            if context_api:
                is_search = False

        if is_search:
            if context_api:
                bestScores = self.tt.LongTensor(self.size).fill_(tempBeamLk[context_api])
                bestScoresId = self.tt.LongTensor(self.size).fill_(context_api)    
            else:
                # print(f"flatBeamLk - here")
                # print(f"beamLk.size: {beamLk.size()}")
                # print(f"beamLk: {beamLk}")
                flatBeamLk = beamLk.view(-1)
                bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)
                # print(f"bestScores(--): {bestScores}")
                # print(f"bestScoresId(--): {bestScoresId}")
        else:
            # print(f"not is_search, context_api: {context_api}")
            bestScores = self.tt.LongTensor(self.size).fill_(tempBeamLk[context_api])
            # bestScores = self.tt.LongTensor(self.size).fill_(1)
            bestScoresId = self.tt.LongTensor(self.size).fill_(context_api)

        # print(f"bestScores: {bestScores}")
        # print(f"bestScoresId: {bestScoresId}")
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        # print(f"self.prevKs: {self.prevKs}")

        self.nextYs.append((bestScoresId - prevK * numWords))
        # print(f"self.nextYs: {self.nextYs}")

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        # print(f"### GETFINAL")
        # print(f"self.finished: {self.finished}")
        # print(f"self.context_api_len: {self.context_api_len}")
        # self.finished = self.finished[self.context_api_len:]

        # print(f"self.finished -filterd: {self.finished}")

        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))

        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
    
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        # print(f"beam-res: {beam_res}")
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]

            if len(hyp)-self.context_api_len > 0:
                # print(f"hyp[::-1]: {hyp[::-1]}")
                hyps.append(hyp[::-1][self.context_api_len:])
            # else:
            #     # print(f"hyp less than context_api_len: {hyp[::-1]}")
            #     # print(f"self.context_api_len: {self.context_api_len}")
            #     hyps.append(hyp[::-1])
            else:
                print(f"<= CONTENT API LEN, hyp: {hyp}")
                hyps.append(hyp[::-1])

        # print(f"hyps: {hyps}")
        # print(f"len(hyps): {len(hyps)}")
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence