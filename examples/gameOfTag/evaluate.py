import yaml



def evaluate(model, test_env, discount_factor, frame_stack_size,
             make_video=False):
    total_reward = 0
    test_env.seed(0)
    initial_frame = test_env.reset()
    frame_stack = FrameStack(
        initial_frame, stack_size=frame_stack_size,
        preprocess_fn=preprocess_frame)
    rendered_frame = test_env.render(mode="rgb_array")
    values, rewards, dones = [], [], []
    if make_video:
        video_writer = cv2.VideoWriter(os.path.join(model.video_dir, "step{}.avi".format(model.step_idx)),
                                       cv2.VideoWriter_fourcc(*"MPEG"), 30,
                                       (rendered_frame.shape[1], rendered_frame.shape[0]))
    while True:
        # Predict action given state: π(a_t | s_t; θ)
        state = frame_stack.get_state()
        action, value = model.predict(
            np.expand_dims(state, axis=0), greedy=False)
        frame, reward, done, _ = test_env.step(action[0])
        rendered_frame = test_env.render(mode="rgb_array")
        total_reward += reward
        dones.append(done)
        values.append(value)
        rewards.append(reward)
        frame_stack.add_frame(frame)
        if make_video:
            video_writer.write(cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR))
        if done:
            break
    if make_video:
        video_writer.release()
    returns = compute_returns(np.transpose([rewards], [1, 0]), [
                              0], np.transpose([dones], [1, 0]), discount_factor)
    value_error = np.mean(np.square(np.array(values) - returns))
    return total_reward, value_error


if __name__ == "__main__":
    with open(r"./smarts.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        print(config)

    # Silence the logs of TF
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'