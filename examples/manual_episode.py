from customer_support_env import Action, CustomerSupportEnv


env = CustomerSupportEnv(default_task_id="easy_password_reset")
obs = env.reset()
print(obs.model_dump_json(indent=2))

sequence = [
    Action(action_type="classify_ticket", ticket_id="T-1001", category="account", priority="high"),
    Action(action_type="assign_ticket", ticket_id="T-1001", assigned_team="frontline"),
    Action(action_type="send_reply", ticket_id="T-1001", content="We initiated your password reset. Please check your email."),
    Action(action_type="close_ticket", ticket_id="T-1001"),
]

for step, action in enumerate(sequence, start=1):
    obs, reward, done, info = env.step(action)
    print(f"step={step} reward={reward.score} done={done} progress={reward.progress}")
    if done:
        print(info)
        break
