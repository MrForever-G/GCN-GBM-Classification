import torch 
from evaluation import eval_model
from utils import graph_tv


def train(model, train_loader, test_loader, loss_func, optimizer, scheduler, args):
    num_epochs = args.num_epochs
    writer = args.writer
    logger = args.logger
    device = args.device
    
    model.to(device)
    round = 0
    for epoch in range(num_epochs):
        model.train()        

        # --- accumulators for sample-weighted logging (reset every epoch) ---
        loss_sum = 0.0          # sum over samples of total loss (cls + lambda_tv * tv)
        clf_sum  = 0.0          # sum over samples of pure classification loss
        tv_sum   = 0.0          # sum over samples of pure TV loss (unweighted)
        n_samples = 0           # total number of samples seen in this epoch

        # --- cache TV hyper-parameters once per epoch (args may not carry these attrs) ---
        lambda_tv = args.lambda_tv
        tv_norm   = args.tv_norm
        tv_reduce = args.tv_reduce
        tv_unique = args.tv_unique

        batch_count = 0
        for data in train_loader:
            
            data = data.to(device)
            y = data.y

            # forward
            y_pred = model(data)

            # classification loss
            classification_loss = loss_func(y_pred, y)

            # TV loss (averaged over available graphs in model.tv_context)
            tv_loss = torch.tensor(0.0, device=device)
            if lambda_tv > 0.0 and hasattr(model, "tv_context") and len(model.tv_context) > 0:
                acc = 0.0
                for node_feat, ei in model.tv_context:
                    acc += graph_tv(
                        node_feat, ei,
                        norm=tv_norm,
                        reduce=tv_reduce,
                        unique=tv_unique,
                    )
                tv_loss = acc / len(model.tv_context)

            # total loss
            total_loss = classification_loss + lambda_tv * tv_loss

            # backward & update (single step on total_loss)
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            # scheduler per-step (keep your current policy)
            if (scheduler is not None) and (args.scheduler not in ["MULTILR", "ExponentialLR"]):
                scheduler.step()
                learning_rate = scheduler.get_last_lr()[0]
                writer.add_scalar("LR", learning_rate, round)
                round += 1

            # sample-weighted aggregation (prefer data.num_graphs for PyG graph classification)
            bs = getattr(data, "num_graphs", None)
            if bs is None:
                bs = y.size(0) if hasattr(y, "size") else 1

            loss_sum += float(total_loss.item()) * bs
            clf_sum  += float(classification_loss.item()) * bs
            tv_sum   += float(tv_loss.item()) * bs
            n_samples += bs

        if args.scheduler == "MULTILR" or args.scheduler == "ExponentialLR":
            scheduler.step()
            learning_rate = scheduler.get_last_lr()[0]
            writer.add_scalar("LR", learning_rate, epoch+1)



        eval_train = eval_model(model, train_loader, device)
        eval_test = eval_model(model, test_loader, device)


        """log"""
        # logger.info("Epoch[%d/%d], loss:%.4f" % (epoch+1, num_epochs, loss_ / batch_count))
        avg_total = loss_sum / max(1, n_samples)
        avg_clf   = clf_sum  / max(1, n_samples)
        avg_tv    = tv_sum   / max(1, n_samples)

        logger.info(
            "Epoch[%d/%d], loss: %.4f (clf: %.4f, tv: %.6f)",
            epoch + 1, num_epochs, avg_total, avg_clf, avg_tv
        )
        logger.info("Epoch[%d/%d], train AUC: %.4f(%.4f-%.4f), test AUC: %.4f(%.4f-%.4f)"
                    % (epoch+1, num_epochs, eval_train[0], eval_train[1], eval_train[2], 
                       eval_test[0], eval_test[1], eval_test[2])) 
        
        """tensorboard"""
        # writer.add_scalar("LOSS", loss_ /batch_count, epoch + 1)
        writer.add_scalar("LOSS/total", avg_total, epoch + 1)
        writer.add_scalar("LOSS/clf",   avg_clf,   epoch + 1)
        writer.add_scalar("LOSS/tv",    avg_tv,    epoch + 1)
        writer.add_scalars("AUC", {"AUC_train": eval_train[0], "AUC_test": eval_test[0]}, epoch + 1)
        
        if (epoch+1) == num_epochs and args.savedisk:
    
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": avg_total,
                "NUM_EPOCHS": num_epochs,
                "DEVICE": device
            }

            torch.save(state, "{}/model.pt".format(args.model_path))
        
