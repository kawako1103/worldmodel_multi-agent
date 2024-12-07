"""
./scenarios/ にリストされているシナリオの1つを使用して、
マルチエージェント環境を作成するためのコード。

たとえば、以下のように呼び出すことで環境を作成できます：
    env = make_env('simple_speaker_listener')

環境オブジェクトを作成した後は、OpenAI Gym 環境と同様に使用できます。

この環境を使用するポリシーは、すべてのエージェントのアクションを
リスト形式で出力する必要があります。リストの各要素は numpy 配列であり、
サイズは (env.world.dim_p + env.world.dim_c, 1) です。
この配列では、物理アクションが通信アクションの前に配置されます。
詳細は environment.py を参照してください。
"""

def make_env(scenario_name, benchmark=False):
    '''
    MultiAgentEnv オブジェクトとして環境を作成します。
    これは、env.reset() と env.step() を呼び出すことで Gym 環境と同様に使用できます。
    env.render() を使用すると、画面上に環境を表示できます。

    入力:
        scenario_name   :   ./scenarios/ にあるシナリオの名前（.py 拡張子を除く）
        benchmark       :   ベンチマークデータを生成するかどうか
                            （通常は評価時にのみ使用）

    有用な環境プロパティ（詳細は environment.py を参照）:
        .observation_space  :   各エージェントの観測空間を返します
        .action_space       :   各エージェントのアクション空間を返します
        .n                  :   エージェント数を返します
    '''
    env = None
    from multiagent.environment import MultiAgentEnv
    print("HELLO FROM CUSTOMIZED MULTIAGENT ENV!")  # デバッグ用の出力

    # 特定のカスタム環境（例: half_cheetah_multi）を処理
    if scenario_name in ["half_cheetah_multi"]:
        if scenario_name == "half_cheetah_multi":
            from multiagent.envs import MultiAgentHalfCheetah
            env = MultiAgentHalfCheetah()
    else:
        import multiagent.scenarios as scenarios

        # スクリプトからシナリオを読み込む この書き方はあまり見かけないが.pyファイルのScenarioクラスをインスタンス化し、オブジェクトを作成している。
        scenario = scenarios.load(scenario_name + ".py").Scenario()

        # ワールドを作成
        world = scenario.make_world()

        # マルチエージェント環境を作成
        if benchmark:
            env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
        else:
            env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env
