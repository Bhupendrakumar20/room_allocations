from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
import pandas as pd
import os
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this in production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOCATION_FOLDER'] = 'allocations'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ALLOCATION_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_csv_structure(df: pd.DataFrame, file_type: str) -> Tuple[bool, str]:
    """Validate CSV structure based on file type"""
    if file_type == 'group':
        required_columns = ['group_id', 'group_name', 'gender', 'size']
        optional_columns = ['preferences', 'special_needs']
    elif file_type == 'hostel':
        required_columns = ['room_id', 'room_name', 'capacity', 'gender_type', 'floor']
        optional_columns = ['amenities', 'room_type']
    else:
        return False, "Invalid file type"
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check for empty required columns
    for col in required_columns:
        if df[col].isnull().any():
            return False, f"Column '{col}' contains empty values"
    
    return True, "Valid structure"

class RoomAllocator:
    def __init__(self):
        self.groups_df = None
        self.rooms_df = None
        self.allocations = []
        self.unallocated_groups = []
        self.allocation_stats = {}
    
    def load_data(self, groups_file: str, rooms_file: str) -> Tuple[bool, str]:
        """Load and validate data from CSV files"""
        try:
            # Load groups data
            self.groups_df = pd.read_csv(groups_file)
            is_valid, message = validate_csv_structure(self.groups_df, 'group')
            if not is_valid:
                return False, f"Groups file error: {message}"
            
            # Load rooms data
            self.rooms_df = pd.read_csv(rooms_file)
            is_valid, message = validate_csv_structure(self.rooms_df, 'hostel')
            if not is_valid:
                return False, f"Rooms file error: {message}"
            
            # Clean and prepare data
            self.groups_df['gender'] = self.groups_df['gender'].str.lower().str.strip()
            self.groups_df['size'] = pd.to_numeric(self.groups_df['size'], errors='coerce')
            
            self.rooms_df['gender_type'] = self.rooms_df['gender_type'].str.lower().str.strip()
            self.rooms_df['capacity'] = pd.to_numeric(self.rooms_df['capacity'], errors='coerce')
            self.rooms_df['available_capacity'] = self.rooms_df['capacity'].copy()
            
            # Validate data integrity
            if self.groups_df['size'].isnull().any():
                return False, "Invalid group sizes found"
            if self.rooms_df['capacity'].isnull().any():
                return False, "Invalid room capacities found"
            
            return True, "Data loaded successfully"
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False, f"Error loading data: {str(e)}"
    
    def allocate_rooms(self) -> Dict:
        """Main allocation algorithm"""
        try:
            self.allocations = []
            self.unallocated_groups = []
            
            # Sort groups by size (descending) for better allocation
            groups_sorted = self.groups_df.sort_values('size', ascending=False)
            
            for _, group in groups_sorted.iterrows():
                allocated = self._allocate_single_group(group)
                if not allocated:
                    self.unallocated_groups.append({
                        'group_id': group['group_id'],
                        'group_name': group['group_name'],
                        'size': group['size'],
                        'gender': group['gender'],
                        'reason': 'No suitable room available'
                    })
            
            # Generate statistics
            self._generate_stats()
            
            return {
                'success': True,
                'allocations': self.allocations,
                'unallocated': self.unallocated_groups,
                'stats': self.allocation_stats
            }
            
        except Exception as e:
            logger.error(f"Error in allocation: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _allocate_single_group(self, group) -> bool:
        """Allocate a single group to available rooms"""
        group_size = int(group['size'])
        group_gender = group['gender']
        
        # Find suitable rooms
        suitable_rooms = self.rooms_df[
            (self.rooms_df['available_capacity'] >= group_size) &
            ((self.rooms_df['gender_type'] == group_gender) | 
             (self.rooms_df['gender_type'] == 'mixed') |
             (group_gender == 'mixed'))
        ].sort_values('available_capacity')  # Prefer smaller suitable rooms
        
        if suitable_rooms.empty:
            return False
        
        # Try to allocate to the best fitting room
        best_room = suitable_rooms.iloc[0]
        room_index = best_room.name
        
        # Update room availability
        self.rooms_df.loc[room_index, 'available_capacity'] -= group_size
        
        # Record allocation
        allocation = {
            'allocation_id': str(uuid.uuid4()),
            'group_id': group['group_id'],
            'group_name': group['group_name'],
            'group_size': group_size,
            'group_gender': group_gender,
            'room_id': best_room['room_id'],
            'room_name': best_room['room_name'],
            'room_capacity': best_room['capacity'],
            'room_gender_type': best_room['gender_type'],
            'floor': best_room['floor'],
            'allocated_at': datetime.now().isoformat()
        }
        self.allocations.append(allocation)
        
        return True
    
    def _generate_stats(self):
        """Generate allocation statistics"""
        total_groups = len(self.groups_df)
        allocated_groups = len(self.allocations)
        unallocated_groups = len(self.unallocated_groups)
        
        total_people = self.groups_df['size'].sum()
        allocated_people = sum(alloc['group_size'] for alloc in self.allocations)
        
        total_rooms = len(self.rooms_df)
        used_rooms = len(set(alloc['room_id'] for alloc in self.allocations))
        
        total_capacity = self.rooms_df['capacity'].sum()
        remaining_capacity = self.rooms_df['available_capacity'].sum()
        
        self.allocation_stats = {
            'total_groups': total_groups,
            'allocated_groups': allocated_groups,
            'unallocated_groups': unallocated_groups,
            'allocation_rate': round((allocated_groups / total_groups) * 100, 2) if total_groups > 0 else 0,
            'total_people': int(total_people),
            'allocated_people': allocated_people,
            'unallocated_people': int(total_people) - allocated_people,
            'total_rooms': total_rooms,
            'used_rooms': used_rooms,
            'room_utilization': round((used_rooms / total_rooms) * 100, 2) if total_rooms > 0 else 0,
            'total_capacity': int(total_capacity),
            'remaining_capacity': int(remaining_capacity),
            'capacity_utilization': round(((total_capacity - remaining_capacity) / total_capacity) * 100, 2) if total_capacity > 0 else 0
        }
    
    def save_results(self, filename: str) -> bool:
        """Save allocation results to CSV"""
        try:
            if not self.allocations:
                return False
            
            df = pd.DataFrame(self.allocations)
            filepath = os.path.join(app.config['ALLOCATION_FOLDER'], filename)
            df.to_csv(filepath, index=False)
            return True
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return False

# Global allocator instance
allocator = RoomAllocator()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads and perform allocation"""
    try:
        # Check if files are present
        if 'group_file' not in request.files or 'hostel_file' not in request.files:
            return jsonify({'success': False, 'error': 'Both files are required'})
        
        group_file = request.files['group_file']
        hostel_file = request.files['hostel_file']
        
        # Validate files
        if group_file.filename == '' or hostel_file.filename == '':
            return jsonify({'success': False, 'error': 'Please select both files'})
        
        if not (allowed_file(group_file.filename) and allowed_file(hostel_file.filename)):
            return jsonify({'success': False, 'error': 'Only CSV files are allowed'})
        
        # Save uploaded files
        group_filename = secure_filename(f"groups_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        hostel_filename = secure_filename(f"hostels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        group_path = os.path.join(app.config['UPLOAD_FOLDER'], group_filename)
        hostel_path = os.path.join(app.config['UPLOAD_FOLDER'], hostel_filename)
        
        group_file.save(group_path)
        hostel_file.save(hostel_path)
        
        # Load data and perform allocation
        success, message = allocator.load_data(group_path, hostel_path)
        if not success:
            return jsonify({'success': False, 'error': message})
        
        # Perform allocation
        result = allocator.allocate_rooms()
        if not result['success']:
            return jsonify({'success': False, 'error': result['error']})
        
        # Save results
        result_filename = f"allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        if allocator.save_results(result_filename):
            result['download_url'] = url_for('download_results', filename=result_filename)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in upload_files: {str(e)}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/download/<filename>')
def download_results(filename):
    """Download allocation results"""
    try:
        filepath = os.path.join(app.config['ALLOCATION_FOLDER'], filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True, download_name=filename)
        else:
            flash('File not found', 'error')
            return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        flash('Error downloading file', 'error')
        return redirect(url_for('index'))

@app.route('/api/stats')
def get_stats():
    """API endpoint for allocation statistics"""
    if hasattr(allocator, 'allocation_stats'):
        return jsonify(allocator.allocation_stats)
    return jsonify({'error': 'No allocation data available'})

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)