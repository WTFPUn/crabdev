import asyncio
import time

async def send_data():
    print("Sending data...")
    await asyncio.sleep(2)  # Simulate sending data every 1 second
    return "eiei"

async def main():
    # Create a task for send_data
    send_data_task = asyncio.create_task(send_data())

    send_data_task.add_done_callback(lambda x: print(f"Task finished! {x.result()}"))

    # Simulate some foreground work (won't block send_data)
    for i in range(5):
        print("Doing some foreground work...")
        await asyncio.sleep(0.5)  # Simulate work taking 0.5 seconds

    # Wait for the task to finish (it won't actually finish in this case)
    # This line is for demonstration purposes only and wouldn't be used in real applications
    await send_data_task

if __name__ == "__main__":
    asyncio.run(main())