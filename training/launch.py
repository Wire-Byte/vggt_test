# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer


def main():
    # 创建命令行参数解析器，用于接收外部传入的配置名。
    # 这样可以通过 --config xxx 在不改代码的情况下切换训练配置文件。
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        # 指定配置文件名称（不带 .yaml 后缀），例如 --config default。
        "--config", 
        type=str, 
        # 默认使用 training/config/default.yaml。
        default="default",
        help="Name of the config file (without .yaml extension, default: default)"
    )
    # 解析命令行参数，得到 args.config。
    args = parser.parse_args()

    # 初始化 Hydra 配置系统。
    # config_path="config" 表示从 training/config 目录中查找配置。
    with initialize(version_base=None, config_path="config"):
        # 组合配置对象：根据传入的配置名加载对应 yaml。
        # 例如 args.config="default" 时会加载 default.yaml。
        #args.config 是从命令行参数中获取的配置文件名（不带 .yaml 后缀）。
        # 例如，如果运行时传入 --config default，Hydra 会加载 default.yaml
        #compose 会将指定的配置文件解析为一个 DictConfig 对象
        cfg = compose(config_name=args.config)

    # 将配置字典解包传入 Trainer，完成训练器实例化。
    # 这里依赖 Trainer 的构造函数参数与配置字段一一对应。
    trainer = Trainer(**cfg)
    # 启动完整训练流程（数据加载、前向反向、日志与保存等）。
    trainer.run()


# 仅当该文件作为主程序执行时才调用 main。
# 若被其他模块 import，则不会自动触发训练。
if __name__ == "__main__":
    main()


