from typing import List
from prompt_generators.prompt import Prompt
from pipeline.run_result import RunResult
import time
from environments.environments import (
    MiniWobEnvironment,
    WebArenaEnvironment,
    OmniactEnvironment,
)
from prompt_generators.prompt_generators import (
    MiniWobPromptGenerator,
    MiniWobSoMPromptGenerator,
    WebArenaSoMPromptGenerator,
    OmniactCotPromptGenerator,
)
from agents.agents import AgentInterface, GPTChatAgent, GPTSoMAgent, OmniactAgent
from response_parsers.miniwob_response_parser import MiniWobResponseParser
from response_parsers.miniwob_som_response_parser import MiniWobSoMResponseParser
from response_parsers.webarena_som_response_parser import WebArenaSoMResponseParser
from utils.config_parser import get_config, ConfigKey
from utils.data_saver import DataSaver
from utils.time_utils import time_function


class AgentPipeline:

    def __init__(self, logger, data_saver: DataSaver) -> None:
        self.logger = logger
        self.data_saver = data_saver

    @time_function
    def get_environment(self, environment_type, environment_name, render_mode="human"):
        if environment_type == "miniwob":
            return MiniWobEnvironment(
                "miniwob/{}-v1".format(environment_name),
                self.logger,
                self.data_saver,
                render_mode=render_mode,
            )
        if environment_type == "webarena" or environment_type == "vwa":
            return WebArenaEnvironment(
                environment_name, self.logger, self.data_saver, render_mode=render_mode
            )
        if environment_type == "omniact":
            return OmniactEnvironment(environment_name, self.logger, self.data_saver)
        else:
            raise Exception("Environment not found")

    @time_function
    def get_prompt_generator(self, prompt_generator_type, env, agent):
        if prompt_generator_type == "miniwob":
            return MiniWobPromptGenerator(env, self.logger)
        elif prompt_generator_type == "miniwob_som":
            return MiniWobSoMPromptGenerator(env, self.logger)
        elif prompt_generator_type == "webarena_som":
            return WebArenaSoMPromptGenerator(env, self.logger)
        elif prompt_generator_type == "omniact_som":
            return OmniactCotPromptGenerator(
                env,
                self.logger,
                self.data_saver,
                get_config(ConfigKey.INSTRUCTION_PATH),
                agent,
            )
        else:
            raise Exception("Prompt Generator not found")

    @time_function
    def get_response_parser(self, response_parser_type, environment):
        if response_parser_type == "miniwob":
            return MiniWobResponseParser(environment)
        elif response_parser_type == "miniwob_som":
            return MiniWobSoMResponseParser(environment)
        elif response_parser_type == "webarena_som":
            return WebArenaSoMResponseParser(environment)
        elif response_parser_type == "omniact_som":
            return None
        else:
            raise Exception("Response Parser not found")

    @time_function
    def get_agent(self, agent_type, environment) -> AgentInterface:
        response_parser = self.get_response_parser(
            get_config(ConfigKey.RESPONSE_PARSER_TYPE), environment
        )
        if agent_type == "gpt_chat":
            return GPTChatAgent(
                get_config(ConfigKey.MODEL_NAME),
                logger=self.logger,
                response_parser=response_parser,
                data_saver=self.data_saver,
            )
        elif agent_type == "gpt_som":
            return GPTSoMAgent(
                get_config(ConfigKey.MODEL_NAME),
                logger=self.logger,
                response_parser=response_parser,
                data_saver=self.data_saver,
            )
        elif agent_type == "omniact":
            return OmniactAgent(
                get_config(ConfigKey.MODEL_NAME),
                logger=self.logger,
                response_parser=response_parser,
                data_saver=self.data_saver,
            )
        else:
            raise Exception("Agent not found")

    def run_pipeline(self, env_name, render_mode="human") -> RunResult:
        run_id = 0
        reward = 0
        while run_id < get_config(ConfigKey.E2E_MAX_RETRIES):
            try:
                run_result = {
                    "reward": 0,
                    "error": False,
                    "error_string": "",
                    "actions": [],
                }
                self.logger.info("Running environment: {}".format(env_name))
                # 1. Construct an Environment
                environment = self.get_environment(
                    get_config(ConfigKey.ENVIRONMENT_TYPE),
                    env_name,
                    render_mode=render_mode,
                )

                self.data_saver.start_run(env_name, run_id, environment.get_task())

                # 2. Construct the Agent
                agent = self.get_agent(
                    get_config(ConfigKey.AGENT_TYPE), environment=environment
                )

                # 3. Construct the Prompt Generators
                prompt_generator = self.get_prompt_generator(
                    get_config(ConfigKey.PROMPT_GENERATOR_TYPE), environment, agent
                )

                # Pipeline Flow Environment -> Task ->  Prompt Generator -> Prompt ->
                # (Agent -> Response -> Response To Action -> Action) -> Environment -> Execution

                # 3. Execute the Agent
                terminated = False
                step_id = 0
                while not terminated and step_id < get_config(ConfigKey.MAX_STEPS):
                    self.data_saver.start_step(step_id)
                    state = environment.get_state()
                    task = environment.get_task()
                    last_step = step_id == get_config(ConfigKey.MAX_STEPS) - 1

                    # Get the prompt
                    prompts: List[Prompt] = prompt_generator.generate_prompts(
                        state,
                        task,
                        agent.prev_responses[-1:],
                        last_step=last_step,
                    )
                    self.data_saver.save_prompts(prompts)

                    # Predict the action
                    if get_config(ConfigKey.AGENT_TYPE) == "omniact":
                        response = agent.predict(prompts, state, task)
                        reward = environment.evaluate_result(response)
                        terminated = True
                    else:
                        action = agent.predict(prompts, state, task)

                        run_result["actions"] = [
                            prev_action.to_json() for prev_action in agent.prev_actions
                        ]
                        reward, terminated = environment.execute_action(action, step_id)

                    self.logger.info("reward: {}".format(reward))
                    # Sleep allows certain transitions (like expanding boxes) to finish)
                    time.sleep(0.5)
                    step_id += 1

                    # Check if previous 3 actions are the same as each other
                    repeat_threshold = get_config(ConfigKey.REPEAT_TOLERANCE)
                    if (
                        len(agent.prev_actions) >= repeat_threshold
                        and len(set(agent.prev_actions[-repeat_threshold:])) == 1
                    ):
                        self.logger.info(
                            "Same action repeated {} times. Terminating episode".format(
                                repeat_threshold
                            )
                        )
                        terminated = True

                # 4. Get the reward
                run_result["reward"] = reward

            except Exception as e:
                self.logger.error("Error running environment: {}".format(env_name))
                self.logger.error("Error: {}".format(e))
                import traceback

                print(traceback.format_exc())
                run_result["error"] = True
                run_result["error_string"] = str(e)
                run_result["defailted_error_string"] = traceback.format_exc()
                self.logger.error("Retrying with retry count: {}".format(run_id))
            finally:
                run_id += 1
                if environment is not None:
                    self.data_saver.save_run_result(run_result)
                    environment.clean_up()
        if run_id >= get_config(ConfigKey.E2E_MAX_RETRIES):
            self.logger.info(
                "Exceeded maximum retries for environment: {}".format(env_name)
            )

        return run_result

    def run_omniact_pipeline(self, env_name, render_mode="human") -> RunResult:
        run_id = 0
        reward = 0
        try:
            run_result = {
                "reward": 0,
                "error": False,
                "error_string": "",
                "actions": [],
            }
            self.logger.info("Running environment: {}".format(env_name))
            # 1. Construct an Environment
            environment = self.get_environment(
                get_config(ConfigKey.ENVIRONMENT_TYPE),
                env_name,
                render_mode=render_mode,
            )

            self.data_saver.start_run(env_name, run_id, environment.get_task())

            # 2. Construct the Agent
            agent = self.get_agent(
                get_config(ConfigKey.AGENT_TYPE), environment=environment
            )

            # 3. Construct the Prompt Generators
            prompt_generator = self.get_prompt_generator(
                get_config(ConfigKey.PROMPT_GENERATOR_TYPE), environment, agent
            )

            # Pipeline Flow Environment -> Task ->  Prompt Generator -> Prompt ->
            # (Agent -> Response -> Response To Action -> Action) -> Environment -> Execution

            # 3. Execute the Agent
            terminated = False
            step_id = 0

            # Save the run_result early
            run_result["gt_json"] = environment.gt_json
            run_result["pred_json"] = [""]
            while not terminated and step_id < get_config(ConfigKey.MAX_STEPS):
                self.data_saver.start_step(step_id)
                state = environment.get_state()
                task = environment.get_task()

                # Get the prompt
                prompts: List[str] = prompt_generator.generate_prompts(state, task)
                self.data_saver.save_prompt_strings(prompts)

                # Predict the action
                if get_config(ConfigKey.AGENT_TYPE) == "omniact":
                    pred_script, response = agent.predict(prompts, state, task)
                    print(response)
                    self.data_saver.save_response(response)
                    # Save the response
                    try:
                        reward = environment.evaluate_result(pred_script)
                    except Exception as e:
                        import traceback

                        print(traceback.format_exc())
                        reward = 0
                        del run_result["gt_json"]
                        del run_result["pred_json"]
                        raise

                    terminated = True
                else:
                    raise Exception("Not omniact agent.")

                self.logger.info("reward: {}".format(reward))
                # Sleep allows certain transitions (like expanding boxes) to finish)
                time.sleep(0.5)
                step_id += 1

                # Check if previous 3 actions are the same as each other
                repeat_threshold = get_config(ConfigKey.REPEAT_TOLERANCE)
                if (
                    len(agent.prev_actions) >= repeat_threshold
                    and len(set(agent.prev_actions[-repeat_threshold:])) == 1
                ):
                    self.logger.info(
                        "Same action repeated {} times. Terminating episode".format(
                            repeat_threshold
                        )
                    )
                    terminated = True

            # 4. Get the reward
            run_result["reward"] = reward

            # 5. save the run jsons
            run_result["pred_json"] = environment.pred_json

        except Exception as e:
            self.logger.error("Error running environment: {}".format(env_name))
            self.logger.error("Error: {}".format(e))
            import traceback

            print(traceback.format_exc())
            run_result["error"] = True
            run_result["error_string"] = str(e)
            run_result["defailted_error_string"] = traceback.format_exc()
            self.logger.error("Retrying with retry count: {}".format(run_id))
        finally:
            run_id += 1
            if environment is not None:
                self.data_saver.save_run_result(run_result)
                environment.clean_up()

        return run_result
