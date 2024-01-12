from accelerate import Accelerator
from omegaconf import open_dict
import hydra
import torch
import time
import os


from .utils import (
    setup_basics,
    train,
    predict,
    eval,
    get_lr_scheduler,
    get_optimizer,
    get_tokenizer,
    get_model,
    get_dataloaders,
    get_config,
)


@hydra.main(config_path="configs", config_name="default", version_base='1.1')
def main(args):
    accelerator = Accelerator(
        cpu=args.device == "cpu",
        mixed_precision=args.precision,
    )
    logger = setup_basics(accelerator, args)
    config = get_config(args)
    model = get_model(args, config)
    tokenizer = get_tokenizer(args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_lr_scheduler(optimizer, args, logger)
    train_dataloader, test_dataloader = get_dataloaders(tokenizer, config, args)

    logger.log_args(args)

    (
        model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        test_dataloader,
    ) = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, test_dataloader
    )

    current_train_step = 1
    resume_training = args.get('accelerator', {}).get('checkpoint_path')
    if resume_training:
        assert os.path.isdir(resume_training) and not resume_training.endswith('/')
        checkpoint_name = os.path.basename(resume_training)
        assert checkpoint_name.startswith('checkpoint-pt-')
        current_train_step = int(checkpoint_name.split('-')[-1])
        # we assume only a single epoch, always
        logger.logger.info(f"Resuming training run from : {resume_training} at training step {current_train_step}")
        # note that if we want to keep using the same out dir we will need to modify the hydra run dir
        accelerator.load_state(input_dir=resume_training)


    if args.model.compile:
        model = torch.compile(model)

    with open_dict(args):
        args.current_train_step = current_train_step
        args.current_epoch = 1
        args.last_log = time.time()

    if args.eval_only:
        model.eval()
        with torch.no_grad():
            eval(model, test_dataloader, logger, args, tokenizer)
    elif args.predict_only:
        model.eval()
        with torch.no_grad():
            predict(model, test_dataloader, logger,
                    args, tokenizer)
    else:
        train(model, train_dataloader, test_dataloader, accelerator,
              lr_scheduler, optimizer, logger, args, tokenizer)

    logger.finish()


if __name__ == "__main__":
    main()
