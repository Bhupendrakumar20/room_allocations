import os
import csv
from flask import Flask, request, render_template, send_from_directory
import pandas as pd

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOCATION_FOLDER = 'allocations'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOCATION_FOLDER'] = ALLOCATION_FOLDER

def allocate_rooms(group_df, hostel_df):
    allocation = []
    group_df.columns = [col.strip() for col in group_df.columns]
    hostel_df.columns = [col.strip() for col in hostel_df.columns]

    group_df['Members'] = group_df['Members'].astype(int)
    hostel_df['Capacity'] = hostel_df['Capacity'].astype(int)

    # Create a dictionary to track room capacities
    room_capacities = {}
    for _, room in hostel_df.iterrows():
        room_capacities[room['Room Number']] = room['Capacity']

    print("Group DataFrame:")
    print(group_df)
    print("Hostel DataFrame:")
    print(hostel_df)

    for _, group in group_df.iterrows():
        group_id = group['Group ID']
        members = group['Members']
        gender = group['Gender']

        allocated = False
        reason = ""

        print(f"Allocating group {group_id} with {members} members ({gender})")

        # Check if the group is mixed gender
        if ' & ' in gender:
            # Split the gender string
            genders = gender.split(' & ')
            # Find the number of each gender
            for i in range(len(genders)):
                genders[i] = genders[i].strip().split(' ')[0]
                genders[i] = int(genders[i])

            # Allocate to the room based on the number of each gender
            for _, room in hostel_df.iterrows():
                room_num = room['Room Number']
                room_gender = room['Gender']

                # If room gender matches and there's enough capacity
                if room_gender == 'Boy' or room_gender == 'Girl':
                    if (room_gender == 'Boy' and genders[0] > 0) or (room_gender == 'Girl' and genders[1] > 0):
                        if room_capacities[room_num] >= members:
                            print(f"Allocating room {room_num} in {room['Hostel Name']} to group {group_id}")

                            allocation.append([group_id, room['Hostel Name'], room_num, members])
                            room_capacities[room_num] -= members

                            # Update gender counts based on allocation
                            if room_gender == 'Boy':
                                genders[0] -= members
                            else:
                                genders[1] -= members

                            # If all genders in the group are allocated
                            if genders[0] == 0 and genders[1] == 0:
                                allocated = True
                                break  # Stop looking for rooms
                        else:
                            print(f"Room {room_num} in {room['Hostel Name']} cannot be allocated to group {group_id} (Insufficient Capacity)")
                else:
                    print(f"Room {room_num} in {room['Hostel Name']} cannot be allocated to group {group_id} (Gender Mismatch)")

            # If any gender from the mixed group remains unallocated
            if not allocated:
                print(f"No suitable room found for group {group_id}")
                reason = "Insufficient capacity in rooms matching gender"
                allocation.append([group_id, "Not allocated", "Not allocated", members, reason])

        else:  # Specific gender (Boy or Girl) allocation
            # Allocate to the room based on the specific gender
            for _, room in hostel_df.iterrows():
                room_num = room['Room Number']
                print(f"Checking room {room_num} in {room['Hostel Name']} with capacity {room_capacities[room_num]} ({room['Gender']})")

                if room['Gender'] == gender and room_capacities[room_num] >= members:
                    print(f"Allocating room {room_num} in {room['Hostel Name']} to group {group_id}")

                    allocation.append([group_id, room['Hostel Name'], room_num, members])
                    room_capacities[room_num] -= members
                    allocated = True
                    break
                else:
                    print(f"Room {room_num} in {room['Hostel Name']} cannot be allocated to group {group_id}")

            if not allocated:
                print(f"No suitable room found for group {group_id}")
                if gender not in hostel_df['Gender'].values:
                    reason = "No rooms matching gender"
                else:
                    reason = "Insufficient capacity in available rooms"
                allocation.append([group_id, "Not allocated", "Not allocated", members, reason])

    allocation_df = pd.DataFrame(allocation, columns=['Group ID', 'Hostel Name', 'Room Number', 'Members Allocated', 'Reason'])
    print("Allocation DataFrame:")
    print(allocation_df)
    return allocation_df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    group_file = request.files['group_file']
    hostel_file = request.files['hostel_file']

    if group_file and hostel_file:
        group_path = os.path.join(app.config['UPLOAD_FOLDER'], 'group_info.csv')
        hostel_path = os.path.join(app.config['UPLOAD_FOLDER'], 'hostel_info.csv')
        group_file.save(group_path)
        hostel_file.save(hostel_path)

        group_df = pd.read_csv(group_path)
        hostel_df = pd.read_csv(hostel_path)

        allocation_df = allocate_rooms(group_df, hostel_df)
        allocation_path = os.path.join(app.config['ALLOCATION_FOLDER'], 'allocation.csv')
        allocation_df.to_csv(allocation_path, index=False)

        return allocation_df.to_html()

@app.route('/download')
def download_file():
    return send_from_directory(app.config['ALLOCATION_FOLDER'], 'allocation.csv')

if __name__ == '__main__':
    app.run(debug=True)