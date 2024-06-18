## peft-test
peft的测试demo，如果配置了wandb，训练日志自动上传


### 环境配置
- 【推荐】使用vscode的`Dev Containers`模式，参考[.devcontainer/README.md](.devcontainer/README.md)

- 【可选】其他虚拟环境方式
    - 【二选一】安装torch-cpu版
        ```shell
        pip install torch torchvision
        ```
    - 【二选一】安装torch-cuda版
        ```shell
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        ```
    - 【必要】安装依赖
        ```shell
        pip install -r requirements.txt
        ```

### Demo
情感分析:
- 训练:
    ```shell
    python emotional_train.py
    ```

- 推测:
    ```shell
    python emotional_test.py
    ```


### 截图
![image](https://github.com/Samge0/peft-test/assets/17336101/7b5c3f60-b8e5-4e76-b278-9290516a3513)