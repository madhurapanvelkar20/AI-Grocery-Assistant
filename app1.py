import pandas as pd
import streamlit as st
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import cv2
import numpy as np
import requests
from PIL import Image
import json
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Grocery Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Email Configuration ---
EMAIL_ADDRESS = "powerbiwork0@gmail.com"  # Replace with your email
EMAIL_PASSWORD = "hgel nxua dwjw zenb"  # Replace with your email password
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465

# --- Load Dataset ---
@st.cache_data
def load_data():
    """Load grocery dataset from CSV file"""
    try:
        df = pd.read_csv("dataset.csv")
        
        # Map your dataset columns to the required format
        column_mapping = {
            'Product Name': 'product_name',
            'Brand': 'brand',
            'Category': 'category',
            'Price (₹)': 'price',
            'Ingredients': 'ingredients',
            'Calories': 'calories',
            'Protein (g)': 'protein_g',
            'Fat (g)': 'fat_g',
            'Carbohydrates (g)': 'carbs_g',
            'Health Tags': 'health_tags',
            'Popularity Score': 'popularity_score',
            'Weight(grams)': 'weight_grams'
        }
        
        # Rename columns to match expected format
        df = df.rename(columns=column_mapping)
        
        # Create missing required columns
        df['product_id'] = range(1, len(df) + 1)  # Generate product IDs
        df['weight_kg'] = df['weight_grams'] / 1000  # Convert grams to kg
        df['barcode'] = df['product_id'].apply(lambda x: f"12345{str(x).zfill(8)}")  # Generate barcodes
        
        # Clean and validate data
        df = df.dropna(subset=['product_name', 'category', 'price'])
        df['health_tags'] = df['health_tags'].fillna('')
        df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce').fillna(0.1)
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
        df['calories'] = pd.to_numeric(df['calories'], errors='coerce').fillna(0)
        df['protein_g'] = pd.to_numeric(df['protein_g'], errors='coerce').fillna(0)
        df['fat_g'] = pd.to_numeric(df['fat_g'], errors='coerce').fillna(0)
        df['carbs_g'] = pd.to_numeric(df['carbs_g'], errors='coerce').fillna(0)
        df['popularity_score'] = pd.to_numeric(df['popularity_score'], errors='coerce').fillna(0)
        
        return df
    except FileNotFoundError:
        st.error("dataset.csv not found. Please make sure the file exists in the same directory.")
        return pd.DataFrame()  # Return empty dataframe
    except Exception as e:
        st.error(f"Error loading dataset.csv: {str(e)}")
        return pd.DataFrame()  # Return empty dataframe

@st.cache_data
def load_offers():
    """Load product offers from CSV file or create sample offers"""
    try:
        offers_df = pd.read_csv("product_offers (1).csv")
        
        # Debug: Show what columns we actually have
       
        
        # Map your actual columns to expected format
        column_mapping = {
            'Product Name': 'product_name',
            'Discounted Price (₹)': 'discounted_price',
            'Discount (%)': 'discount_percentage'
        }
        
        # Rename columns
        offers_df = offers_df.rename(columns=column_mapping)
        
        # Load main dataset to get product IDs and original prices
        main_df = load_data()
        if not main_df.empty:
            # Merge with main dataset to get product_id and original price
            offers_df = offers_df.merge(
                main_df[['product_id', 'product_name', 'price']], 
                on='product_name', 
                how='left'
            )
            
            # Rename price to original_price for consistency
            offers_df = offers_df.rename(columns={'price': 'original_price'})
            
            # Add a default offer end date (30 days from now)
            offers_df['offer_ends_on'] = pd.to_datetime('today') + pd.Timedelta(days=30)
            
            # Remove rows where product couldn't be matched
            offers_df = offers_df.dropna(subset=['product_id'])
            
            # Clean up the dataframe
            offers_df = offers_df[['product_id', 'product_name', 'original_price', 'discounted_price', 'discount_percentage', 'offer_ends_on']]
            
            st.success(f" Exciting Offers available")
            return offers_df
        else:
            st.warning("No Offers available")
            return create_sample_offers()
            
    except FileNotFoundError:
        st.info("product_offers (1).csv not found. Creating sample offers based on your dataset.")
        return create_sample_offers()
    except Exception as e:
        st.error(f"Error loading product_offers (1).csv: {str(e)}. Creating sample offers.")
        st.error(f"Make sure your offers file has columns: Product Name, Discounted Price (₹), Discount (%)")
        return create_sample_offers()

def create_sample_offers():
    """Create sample offers based on the main dataset"""
    df = load_data()
    if df.empty:
        return pd.DataFrame()
    
    # Create offers for first 10 products as sample
    sample_products = df.head(10)
    
    offers_data = {
        'product_id': sample_products['product_id'].tolist(),
        'original_price': sample_products['price'].tolist(),
        'discounted_price': (sample_products['price'] * 0.8).tolist(),  # 20% discount
        'discount_percentage': [20] * len(sample_products),
        'offer_ends_on': [(datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(len(sample_products))]
    }
    
    offers_df = pd.DataFrame(offers_data)
    offers_df['offer_ends_on'] = pd.to_datetime(offers_df['offer_ends_on'])
    return offers_df

# --- Barcode Validation Functions ---
def validate_barcode(barcode_str):
    """Validate barcode format and check digit"""
    if not str(barcode_str).isdigit():
        return False
    
    # EAN-13 validation
    if len(str(barcode_str)) == 13:
        return validate_ean13(str(barcode_str))
    
    # EAN-8 validation
    if len(str(barcode_str)) == 8:
        return validate_ean8(str(barcode_str))
    
    # UPC-A validation
    if len(str(barcode_str)) == 12:
        return validate_upca(str(barcode_str))
    
    # Basic length check for other formats
    return len(str(barcode_str)) >= 8

def validate_ean13(barcode):
    """Validate EAN-13 barcode check digit"""
    if len(barcode) != 13:
        return False
    
    try:
        odd_sum = sum(int(barcode[i]) for i in range(0, 12, 2))
        even_sum = sum(int(barcode[i]) for i in range(1, 12, 2))
        
        total = odd_sum + (even_sum * 3)
        check_digit = (10 - (total % 10)) % 10
        
        return check_digit == int(barcode[12])
    except:
        return False

def validate_ean8(barcode):
    """Validate EAN-8 barcode check digit"""
    if len(barcode) != 8:
        return False
    
    try:
        odd_sum = sum(int(barcode[i]) for i in range(0, 7, 2))
        even_sum = sum(int(barcode[i]) for i in range(1, 7, 2))
        
        total = (odd_sum * 3) + even_sum
        check_digit = (10 - (total % 10)) % 10
        
        return check_digit == int(barcode[7])
    except:
        return False

def validate_upca(barcode):
    """Validate UPC-A barcode check digit"""
    if len(barcode) != 12:
        return False
    
    try:
        odd_sum = sum(int(barcode[i]) for i in range(0, 11, 2))
        even_sum = sum(int(barcode[i]) for i in range(1, 11, 2))
        
        total = (odd_sum * 3) + even_sum
        check_digit = (10 - (total % 10)) % 10
        
        return check_digit == int(barcode[11])
    except:
        return False

# --- OpenCV Barcode Region Detection ---
def detect_barcode_region(image):
    """Detect potential barcode regions using OpenCV"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        
        # Calculate gradient magnitude
        gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        
        # Subtract y-gradient from x-gradient
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        
        # Apply threshold and morphological operations
        blurred = cv2.blur(gradient, (9, 9))
        (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
        
        # Construct closing kernel and apply
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Perform erosions and dilations
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)
        
        # Find contours
        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Check if we found potential barcode regions
        if contours:
            # Get the largest contour (likely the barcode)
            largest_contour = contours[0]
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Check if the dimensions are reasonable for a barcode
            if w > h and w > 50 and h > 10:
                return {
                    'detected': True,
                    'region': (x, y, w, h),
                    'confidence': cv2.contourArea(largest_contour),
                    'processed_image': closed
                }
        
        return {'detected': False, 'region': None, 'confidence': 0, 'processed_image': None}
    
    except Exception as e:
        st.error(f"Error in barcode region detection: {str(e)}")
        return {'detected': False, 'region': None, 'confidence': 0, 'processed_image': None}

# --- Simulate barcode scanning from your dataset ---
def simulate_barcode_scan(barcode_input):
    """Simulate barcode scanning by looking up product in your dataset"""
    df = load_data()
    
    if df.empty:
        return {'found': False}
    
    # Try to find product by barcode
    product = df[df['barcode'] == str(barcode_input)]
    
    if not product.empty:
        product_data = product.iloc[0]
        return {
            'found': True,
            'product_id': product_data['product_id'],
            'name': product_data['product_name'],
            'brand': product_data.get('brand', 'Unknown'),
            'category': product_data['category'],
            'price': product_data['price'],
            'weight_kg': product_data['weight_kg'],
            'calories': product_data.get('calories', 0),
            'protein_g': product_data.get('protein_g', 0),
            'fat_g': product_data.get('fat_g', 0),
            'carbs_g': product_data.get('carbs_g', 0),
            'health_tags': product_data['health_tags'].split(',') if pd.notna(product_data['health_tags']) else [],
            'ingredients': product_data.get('ingredients', ''),
            'popularity_score': product_data.get('popularity_score', 0)
        }
    else:
        return {'found': False}

# --- Health Assessment ---
def assess_health_for_customer(product_info, customer_profile):
    """Assess if product is good for customer's health based on their profile"""
    if not product_info or not product_info.get('found'):
        return {"score": 0, "warnings": ["Product information not available"], "recommendations": []}
    
    health_score = 70  # Base score
    warnings = []
    recommendations = []
    
    # Get customer conditions
    conditions = customer_profile.get('conditions', [])
    age = customer_profile.get('age', 30)
    activity_level = customer_profile.get('activity_level', 'moderate')
    allergies = customer_profile.get('allergies', [])
    
    # Health tags assessment
    health_tags = product_info.get('health_tags', [])
    
    # Positive health factors
    if any('protein' in tag.lower() for tag in health_tags):
        health_score += 10
        recommendations.append("✅ Good protein source")
    
    if any('fiber' in tag.lower() for tag in health_tags):
        health_score += 10
        recommendations.append("✅ High fiber content")
    
    if any('organic' in tag.lower() for tag in health_tags):
        health_score += 5
        recommendations.append("✅ Organic product")
    
    # Nutritional assessment based on your dataset
    calories = product_info.get('calories', 0)
    protein_g = product_info.get('protein_g', 0)
    fat_g = product_info.get('fat_g', 0)
    carbs_g = product_info.get('carbs_g', 0)
    
    # Category-based assessment
    category = product_info.get('category', '')
    if category.lower() in ['fruits', 'vegetables']:
        health_score += 15
        recommendations.append("✅ Excellent choice - fruits/vegetables")
    elif category.lower() in ['nuts', 'grains']:
        health_score += 10
        recommendations.append("✅ Good choice - nutritious category")
    elif category.lower() in ['snacks']:
        health_score -= 10
        warnings.append("⚠️ Processed snack - consume in moderation")
    
    # Health condition specific checks
    if 'diabetes' in conditions:
        if carbs_g > 15:
            health_score -= 15
            warnings.append("⚠️ High carbohydrate content - monitor blood sugar")
        elif carbs_g > 5:
            warnings.append("⚠️ Moderate carbs - consume in moderation")
    
    if 'hypertension' in conditions:
        if category.lower() in ['meat', 'snacks']:
            warnings.append("⚠️ Check sodium content")
            health_score -= 5
    
    if 'weight_management' in conditions:
        if calories > 400:
            health_score -= 15
            warnings.append("⚠️ High calorie - watch portion size")
        elif category.lower() in ['fruits', 'vegetables']:
            health_score += 10
            recommendations.append("✅ Great for weight management")
    
    # Age-based recommendations
    if age > 65:
        if protein_g > 10:
            health_score += 10
            recommendations.append("✅ Good protein for muscle maintenance")
    
    # Activity level adjustments
    if activity_level == 'high':
        if protein_g > 15:
            health_score += 10
            recommendations.append("✅ Excellent protein for active lifestyle")
        if category.lower() == 'grains':
            health_score += 5
            recommendations.append("✅ Good carb source for energy")
    
    # Check for potential allergens
    product_name = product_info.get('name', '').lower()
    ingredients = product_info.get('ingredients', '').lower()
    
    for allergy in allergies:
        if allergy.lower() in product_name or allergy.lower() in ingredients:
            health_score -= 50
            warnings.append(f"🚨 May contain {allergy} - AVOID if allergic")
    
    # Ensure score is within bounds
    health_score = max(0, min(100, health_score))
    
    return {
        "score": health_score,
        "warnings": warnings,
        "recommendations": recommendations
    }

# --- Email Functionality ---
def send_email(to_email, subject, body):
    """Send email notification"""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_ADDRESS, to_email, text)
        server.quit()
        
        return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False

# --- Helper Functions ---
def add_to_shopping_list(product_info, quantity=1):
    """Add product to shopping list"""
    if product_info and product_info.get('found'):
        item = {
            'product_id': product_info['product_id'],
            'name': product_info['name'],
            'brand': product_info.get('brand', 'Unknown'),
            'category': product_info['category'],
            'price': product_info['price'],
            'weight_kg': product_info['weight_kg'],
            'quantity': quantity,
            'total_price': product_info['price'] * quantity,
            'total_weight': product_info['weight_kg'] * quantity,
            'health_tags': product_info.get('health_tags', []),
            'calories': product_info.get('calories', 0),
            'protein_g': product_info.get('protein_g', 0),
            'added_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Check if item already exists
        existing_item = next((i for i, x in enumerate(st.session_state.shopping_list) if x['product_id'] == item['product_id']), None)
        
        if existing_item is not None:
            st.session_state.shopping_list[existing_item]['quantity'] += quantity
            st.session_state.shopping_list[existing_item]['total_price'] += item['total_price']
            st.session_state.shopping_list[existing_item]['total_weight'] += item['total_weight']
        else:
            st.session_state.shopping_list.append(item)
        
        # Update totals
        st.session_state.total_weight += item['total_weight']
        st.session_state.total_cost += item['total_price']
        
        return True
    return False

def remove_from_shopping_list(index):
    """Remove item from shopping list"""
    if 0 <= index < len(st.session_state.shopping_list):
        item = st.session_state.shopping_list[index]
        st.session_state.total_weight -= item['total_weight']
        st.session_state.total_cost -= item['total_price']
        st.session_state.shopping_list.pop(index)

def add_notification(message, type="info"):
    """Add notification to session state"""
    notification = {
        'message': message,
        'type': type,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    st.session_state.notifications.append(notification)

# --- Initialize Session State ---
if 'shopping_list' not in st.session_state:
    st.session_state.shopping_list = []
if 'customer_profile' not in st.session_state:
    st.session_state.customer_profile = {
        'name': '',
        'age': 30,
        'conditions': [],
        'allergies': [],
        'activity_level': 'moderate',
        'email': ''
    }
if 'scanned_products' not in st.session_state:
    st.session_state.scanned_products = []
if 'total_weight' not in st.session_state:
    st.session_state.total_weight = 0
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0
if 'notifications' not in st.session_state:
    st.session_state.notifications = []

# --- Sidebar Configuration ---
st.sidebar.title("🛒 Smart Grocery Assistant")
st.sidebar.markdown("---")

# Customer Profile Section
st.sidebar.subheader("👤 Customer Profile")
with st.sidebar.expander("Profile Settings", expanded=True):
    st.session_state.customer_profile['name'] = st.text_input("Name", value=st.session_state.customer_profile['name'])
    st.session_state.customer_profile['age'] = st.number_input("Age", min_value=1, max_value=120, value=st.session_state.customer_profile['age'])
    st.session_state.customer_profile['email'] = st.text_input("Email", value=st.session_state.customer_profile['email'])
    
    # Health conditions
    health_conditions = st.multiselect(
        "Health Conditions",
        ["diabetes", "hypertension", "heart_disease", "weight_management", "celiac_disease"],
        default=st.session_state.customer_profile['conditions']
    )
    st.session_state.customer_profile['conditions'] = health_conditions
    
    # Allergies
    allergies = st.multiselect(
        "Allergies",
        ["milk", "eggs", "peanuts", "tree nuts", "soy", "wheat", "fish", "shellfish"],
        default=st.session_state.customer_profile['allergies']
    )
    st.session_state.customer_profile['allergies'] = allergies
    
    # Activity level
    st.session_state.customer_profile['activity_level'] = st.selectbox(
        "Activity Level",
        ["low", "moderate", "high"],
        index=["low", "moderate", "high"].index(st.session_state.customer_profile['activity_level'])
    )

# Quick Stats in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Quick Stats")
st.sidebar.metric("Shopping List Items", len(st.session_state.shopping_list))
st.sidebar.metric("Total Cost", f"₹{st.session_state.total_cost:.2f}")
st.sidebar.metric("Total Weight", f"{st.session_state.total_weight:.2f} kg")

# --- Main Application ---
st.title("🛒 Smart Grocery Assistant")
st.markdown("Scan barcodes, get health insights, and manage your shopping list intelligently!")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Home Dashboard", "🔍 Barcode Scanner", "📋 Shopping List", "📊 Health Dashboard", "📧 Notifications"])

# Tab 1: Home Dashboard
with tab1:
    st.header("🏠 Home Dashboard")
    
    # Welcome message
    if st.session_state.customer_profile['name']:
        st.markdown(f"Welcome back, **{st.session_state.customer_profile['name']}**! 👋")
    else:
        st.markdown("Welcome to Smart Grocery Assistant! Please set up your profile in the sidebar.")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Scanned Products", len(st.session_state.scanned_products))
    with col2:
        st.metric("Shopping List Items", len(st.session_state.shopping_list))
    with col3:
        st.metric("Total Weight (kg)", f"{st.session_state.total_weight:.2f}")
    with col4:
        if st.session_state.scanned_products:
            avg_health_score = sum([p.get('health_score', 0) for p in st.session_state.scanned_products]) / len(st.session_state.scanned_products)
            st.metric("Avg Health Score", f"{avg_health_score:.1f}%")
        else:
            st.metric("Avg Health Score", "0%")
    
    # Product Discovery
    st.subheader("🔍 Product Discovery")
    
    # Load products from your dataset
    df = load_data()
    offers_df = load_offers()
    
    if not df.empty:
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            categories = ["All"] + sorted(df['category'].unique().tolist())
            selected_category = st.selectbox("Select Category", categories)
        
        with col2:
            health_filters = ["All", "High Protein", "High Fiber", "Organic", "Low Calorie"]
            selected_health_filter = st.selectbox("Select Health Filter", health_filters)
        
        # Filter products
        filtered_df = df.copy()
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        
        if selected_health_filter != "All":
            filter_map = {
                "High Protein": "protein",
                "High Fiber": "fiber",
                "Organic": "organic",
                "Low Calorie": "low calorie"
            }
            filter_tag = filter_map.get(selected_health_filter, "")
            if filter_tag:
                filtered_df = filtered_df[filtered_df['health_tags'].str.contains(filter_tag, case=False, na=False)]
        
        # Display products
        st.subheader("Available Products")
        
        if not filtered_df.empty:
            # Display products in a grid
            products_per_row = 3
            for i in range(0, min(len(filtered_df), 15), products_per_row):  # Show max 15 products
                cols = st.columns(products_per_row)
                for j, (idx, product) in enumerate(filtered_df.iloc[i:i+products_per_row].iterrows()):
                    if j < len(cols):
                        with cols[j]:
                            st.subheader(product['product_name'])
                            if 'brand' in product and pd.notna(product['brand']):
                                st.write(f"**Brand:** {product['brand']}")
                            st.write(f"**Category:** {product['category']}")
                            st.write(f"**Price:** ₹{product['price']:.2f}")
                            st.write(f"**Weight:** {product['weight_kg']:.3f} kg")
                            
                            # Show nutritional info if available
                            if 'calories' in product and product['calories'] > 0:
                                st.write(f"**Calories:** {product['calories']}")
                            
                            # Check for offers
                            offer = offers_df[offers_df['product_id'] == product['product_id']]
                            if not offer.empty:
                                discount = offer.iloc[0]['discount_percentage']
                                discounted_price = offer.iloc[0]['discounted_price']
                                st.success(f"🎉 {discount}% OFF!")
                                st.write(f"**Sale Price:** ₹{discounted_price:.2f}")
                            
                            # Health tags
                            if pd.notna(product['health_tags']):
                                tags = product['health_tags'].split(',')
                                st.write("**Health Benefits:**")
                                for tag in tags[:3]:  # Show first 3 tags
                                    st.write(f"• {tag.strip()}")
                            
                            # Add to shopping list button
                            if st.button(f"Add to Cart", key=f"add_{product['product_id']}"):
                                product_info = {
                                    'found': True,
                                    'product_id': product['product_id'],
                                    'name': product['product_name'],
                                    'brand': product.get('brand', 'Unknown'),
                                    'category': product['category'],
                                    'price': product['price'],
                                    'weight_kg': product['weight_kg'],
                                    'health_tags': product['health_tags'].split(',') if pd.notna(product['health_tags']) else [],
                                    'calories': product.get('calories', 0),
                                    'protein_g': product.get('protein_g', 0),
                                    'fat_g': product.get('fat_g', 0),
                                    'carbs_g': product.get('carbs_g', 0)
                                }
                                if add_to_shopping_list(product_info):
                                    st.success(f"Added {product['product_name']} to cart!")
                                    add_notification(f"Added {product['product_name']} to shopping list", "success")
                                    time.sleep(1)
                                    st.rerun()
                            
                            st.markdown("---")
        else:
            st.info("No products found matching your criteria.")
    else:
        st.error("No products loaded from dataset. Please check your dataset.csv file.")

# Tab 2: Barcode Scanner
with tab2:
    st.header("📱 Barcode Scanner")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Scan Methods")
        
        # Method 1: Manual barcode entry
        st.markdown("**Method 1: Manual Entry**")
        barcode_input = st.text_input("Enter barcode manually:", placeholder="e.g., 1234500001")
        
        # Method 2: Image upload
        st.markdown("**Method 2: Upload Image**")
        uploaded_file = st.file_uploader("Upload barcode image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Detect barcode region
            with st.spinner("Detecting barcode region..."):
                detection_result = detect_barcode_region(image)
                
                if detection_result['detected']:
                    st.success(f"Barcode region detected! Confidence: {detection_result['confidence']:.0f}")
                    
                    # Show processed image
                    if detection_result['processed_image'] is not None:
                        st.image(detection_result['processed_image'], caption="Processed Image", use_column_width=True)
                    
                    st.info("Please enter the barcode numbers from the detected region above:")
                    detected_barcode = st.text_input("Detected barcode:", key="detected_barcode")
                    if detected_barcode:
                        barcode_input = detected_barcode
                else:
                    st.warning("No barcode region detected. Please try a clearer image or enter manually.")
    
    with col2:
        st.subheader("Scan Actions")
        
        # Process barcode
        if st.button("🔍 Scan Barcode", type="primary"):
            if barcode_input:
                if validate_barcode(barcode_input):
                    with st.spinner("Looking up product in database..."):
                        product_info = simulate_barcode_scan(barcode_input)
                        
                        if product_info['found']:
                            # Assess health
                            health_assessment = assess_health_for_customer(product_info, st.session_state.customer_profile)
                            
                            # Add to scanned products
                            scan_result = {
                                'barcode': barcode_input,
                                'product_info': product_info,
                                'health_assessment': health_assessment,
                                'health_score': health_assessment['score'],
                                'scan_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            st.session_state.scanned_products.append(scan_result)
                            st.success("Product scanned successfully!")
                            add_notification(f"Scanned {product_info['name']}", "success")
                            st.rerun()
                        else:
                            st.error("Product not found in database.")
                            st.info("Try scanning a different barcode or check our product catalog.")
                else:
                    st.error("Invalid barcode format.")
            else:
                st.warning("Please enter a barcode.")
    
    # Display scanned products
    if st.session_state.scanned_products:
        st.subheader("📦 Recently Scanned Products")
        
        for idx, scan in enumerate(reversed(st.session_state.scanned_products[-5:])):  # Show last 5
            with st.expander(f"{scan['product_info']['name']} - Health Score: {scan['health_score']:.1f}%"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Brand:** {scan['product_info']['brand']}")
                    st.markdown(f"**Category:** {scan['product_info']['category']}")
                    st.markdown(f"**Price:** ₹{scan['product_info']['price']:.2f}")
                    st.markdown(f"**Scanned:** {scan['scan_time']}")
                    
                    # Ingredients if available
                    if scan['product_info'].get('ingredients'):
                        st.markdown(f"**Ingredients:** {scan['product_info']['ingredients'][:100]}...")
                
                with col2:
                    # Nutrition info from your dataset
                    st.markdown("**Nutrition Info:**")
                    st.markdown(f"• Calories: {scan['product_info']['calories']}")
                    st.markdown(f"• Proteins: {scan['product_info']['protein_g']}g")
                    st.markdown(f"• Carbs: {scan['product_info']['carbs_g']}g")
                    st.markdown(f"• Fat: {scan['product_info']['fat_g']}g")
                    
                    # Health tags
                    if scan['product_info']['health_tags']:
                        st.markdown("**Health Tags:**")
                        for tag in scan['product_info']['health_tags'][:3]:
                            st.markdown(f"• {tag.strip()}")
                
                # Health warnings
                if scan['health_assessment']['warnings']:
                    st.markdown("**Health Warnings:**")
                    for warning in scan['health_assessment']['warnings']:
                        st.markdown(f"• {warning}")
                
                # Recommendations
                if scan['health_assessment']['recommendations']:
                    st.markdown("**Recommendations:**")
                    for rec in scan['health_assessment']['recommendations']:
                        st.markdown(f"• {rec}")
                
                # Add to shopping list
                col1, col2 = st.columns(2)
                with col1:
                    quantity = st.number_input("Quantity:", min_value=1, max_value=50, value=1, key=f"qty_scan_{idx}")
                with col2:
                    if st.button(f"Add to Shopping List", key=f"add_scanned_{idx}"):
                        if add_to_shopping_list(scan['product_info'], quantity):
                            st.success("Added to shopping list!")
                            add_notification(f"Added {scan['product_info']['name']} to shopping list", "success")
                            st.rerun()

# Tab 3: Shopping List
with tab3:
    st.header("📋 Shopping List Manager")
    
    # Add manual item
    st.subheader("➕ Add Item from Catalog")
    
    # Load products from dataset
    df = load_data()
    
    if not df.empty:
        # Category filter and product selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Get unique categories from your dataset + "All" option
            available_categories = ["All"] + sorted(df['category'].unique().tolist())
            selected_category = st.selectbox("Filter by Category", available_categories)
        
        with col2:
            # Filter products based on selected category
            if selected_category == "All":
                filtered_products = df
            else:
                filtered_products = df[df['category'] == selected_category]
            
            # Create a dropdown with product names
            if not filtered_products.empty:
                product_options = ["Select a product..."] + filtered_products['product_name'].tolist()
                selected_product_name = st.selectbox("Select Product", product_options)
            else:
                st.warning(f"No products found in {selected_category} category")
                selected_product_name = "Select a product..."
        
        with col3:
            manual_quantity = st.number_input("Quantity", min_value=1, value=1)
        
        if st.button("Add Item"):
            if selected_product_name != "Select a product...":
                # Find the selected product in the dataset
                selected_product = filtered_products[filtered_products['product_name'] == selected_product_name].iloc[0]
                
                product_info = {
                    'found': True,
                    'product_id': selected_product['product_id'],
                    'name': selected_product['product_name'],
                    'brand': selected_product.get('brand', 'Unknown'),
                    'category': selected_product['category'],
                    'price': selected_product['price'],
                    'weight_kg': selected_product['weight_kg'],
                    'health_tags': selected_product['health_tags'].split(',') if pd.notna(selected_product['health_tags']) else [],
                    'calories': selected_product.get('calories', 0),
                    'protein_g': selected_product.get('protein_g', 0),
                    'fat_g': selected_product.get('fat_g', 0),
                    'carbs_g': selected_product.get('carbs_g', 0)
                }
                
                if add_to_shopping_list(product_info, manual_quantity):
                    st.success(f"Added {manual_quantity}x {selected_product_name} to shopping list! Price: ₹{selected_product['price']:.2f}")
                    add_notification(f"Added {selected_product_name} to shopping list", "success")
                    st.rerun()
            else:
                st.warning("Please select a product from the catalog.")
        
        # Show filtered products preview
        if selected_category != "All" and not filtered_products.empty:
            st.markdown(f"**{len(filtered_products)} products available in {selected_category}:**")
            
            # Display first few products as preview
            preview_products = filtered_products.head(5)
            for _, product in preview_products.iterrows():
                st.write(f"• {product['product_name']} - ₹{product['price']:.2f}")
            
            if len(filtered_products) > 5:
                st.write(f"... and {len(filtered_products) - 5} more products")
    
    else:
        st.error("No products loaded from dataset. Please check your dataset.csv file.")
    
    # Display shopping list
    if st.session_state.shopping_list:
        st.subheader("🛒 Your Shopping List")
        
        # Calculate totals
        total_items = sum([item['quantity'] for item in st.session_state.shopping_list])
        total_price = sum([item.get('total_price', 0) for item in st.session_state.shopping_list])
        
        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Items", total_items)
        with col2:
            st.metric("Total Price", f"₹{total_price:.2f}")
        with col3:
            avg_health = sum([item.get('health_score', 50) for item in st.session_state.shopping_list]) / len(st.session_state.shopping_list)
            st.metric("Avg Health Score", f"{avg_health:.1f}%")
        
        # Shopping list table
        for idx, item in enumerate(st.session_state.shopping_list):
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**{item['name']}**")
                    if 'brand' in item and item['brand'] != 'Manual Entry':
                        st.markdown(f"*{item['brand']}*")
                    st.markdown(f"Category: {item['category']}")
                    
                    # Display health info if available
                    if item.get('calories', 0) > 0:
                        st.markdown(f"Calories: {item['calories']} | Protein: {item.get('protein_g', 0)}g")
                
                with col2:
                    new_quantity = st.number_input("Qty", min_value=1, value=item['quantity'], key=f"qty_{idx}")
                    if new_quantity != item['quantity']:
                        # Update quantities and totals
                        old_total_price = item['total_price']
                        old_total_weight = item['total_weight']
                        
                        st.session_state.shopping_list[idx]['quantity'] = new_quantity
                        st.session_state.shopping_list[idx]['total_price'] = item['price'] * new_quantity
                        st.session_state.shopping_list[idx]['total_weight'] = item['weight_kg'] * new_quantity
                        
                        # Update session totals
                        st.session_state.total_cost = st.session_state.total_cost - old_total_price + st.session_state.shopping_list[idx]['total_price']
                        st.session_state.total_weight = st.session_state.total_weight - old_total_weight + st.session_state.shopping_list[idx]['total_weight']
                        
                        st.rerun()
                
                with col3:
                    st.markdown(f"₹{item['price']:.2f}")
                
                with col4:
                    st.markdown(f"₹{item.get('total_price', 0):.2f}")
                
                with col5:
                    if st.button("🗑️", key=f"delete_{idx}", help="Remove item"):
                        remove_from_shopping_list(idx)
                        add_notification(f"Removed {item['name']} from shopping list", "info")
                        st.rerun()
                
                st.markdown("---")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🛒 Checkout", type="primary"):
                st.success("🎉 Thank you for shopping with us!")
                st.balloons()
                
                # Save checkout details
                checkout_summary = {
                    'items': len(st.session_state.shopping_list),
                    'total_cost': st.session_state.total_cost,
                    'total_weight': st.session_state.total_weight,
                    'checkout_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                add_notification(f"Checkout completed! Total: ₹{checkout_summary['total_cost']:.2f}", "success")
                
                # Clear shopping list after checkout
                st.session_state.shopping_list = []
                st.session_state.total_cost = 0
                st.session_state.total_weight = 0
                
                time.sleep(3)
                st.rerun()
        
        with col2:
            if st.button("📧 Email List"):
                if st.session_state.customer_profile['email']:
                    # Create email content
                    email_content = f"""
Shopping List for {st.session_state.customer_profile['name']}
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Items:
"""
                    for item in st.session_state.shopping_list:
                        email_content += f"• {item['name']} - Qty: {item['quantity']} - ₹{item.get('total_price', 0):.2f}\n"
                    
                    email_content += f"""
Total Items: {total_items}
Total Price: ₹{total_price:.2f}
Average Health Score: {avg_health:.1f}%

Generated by Smart Grocery Assistant
"""
                    
                    if send_email(
                        st.session_state.customer_profile['email'],
                        "Your Shopping List",
                        email_content
                    ):
                        st.success("Shopping list sent to your email!")
                        add_notification("Shopping list emailed successfully", "success")
                    else:
                        st.error("Failed to send email. Please check your email settings.")
                else:
                    st.error("Please set your email in the profile settings.")
        
        with col3:
            if st.button("🗑️ Clear All"):
                st.session_state.shopping_list = []
                st.session_state.total_cost = 0
                st.session_state.total_weight = 0
                add_notification("Cleared all items from shopping list", "info")
                st.rerun()
        
        # Category breakdown
        st.markdown("---")
        st.subheader("📊 Category Breakdown")
        
        # Create category summary
        category_summary = {}
        for item in st.session_state.shopping_list:
            category = item['category']
            if category not in category_summary:
                category_summary[category] = {'count': 0, 'total_cost': 0, 'total_weight': 0}
            
            category_summary[category]['count'] += item['quantity']
            category_summary[category]['total_cost'] += item.get('total_price', 0)
            category_summary[category]['total_weight'] += item.get('total_weight', 0)
        
        # Display category breakdown
        for category, data in category_summary.items():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"**{category}**")
            with col2:
                st.write(f"Items: {data['count']}")
            with col3:
                st.write(f"Cost: ₹{data['total_cost']:.2f}")
            with col4:
                st.write(f"Weight: {data['total_weight']:.2f} kg")
        
        # Visualizations
        if category_summary:
            st.markdown("---")
            st.subheader("📈 Shopping Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Cost by category pie chart
                categories = list(category_summary.keys())
                costs = [category_summary[cat]['total_cost'] for cat in categories]
                
                fig_pie = px.pie(
                    values=costs,
                    names=categories,
                    title="Cost Distribution by Category"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Items by category bar chart
                items = [category_summary[cat]['count'] for cat in categories]
                
                fig_bar = px.bar(
                    x=categories,
                    y=items,
                    title="Items by Category",
                    labels={'x': 'Category', 'y': 'Number of Items'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
    
    else:
        st.info("🛒 Your shopping list is empty!")
        st.write("Start by scanning barcodes or browsing products in the Home Dashboard.")

# Tab 4: Health Dashboard
with tab4:
    st.header("📊 Health Dashboard")
    
    if st.session_state.scanned_products:
        # Health score distribution
        health_scores = [p['health_score'] for p in st.session_state.scanned_products]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Health score histogram
            fig_hist = px.histogram(
                x=health_scores,
                nbins=10,
                title="Health Score Distribution",
                labels={'x': 'Health Score', 'y': 'Number of Products'},
                color_discrete_sequence=['#2E86AB']
            )
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Health score over time
            scan_times = [datetime.strptime(p['scan_time'], "%Y-%m-%d %H:%M:%S") for p in st.session_state.scanned_products]
            
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=scan_times,
                y=health_scores,
                mode='lines+markers',
                name='Health Score',
                line=dict(color='#A23B72', width=2),
                marker=dict(size=8)
            ))
            fig_line.update_layout(
                title="Health Score Trends",
                xaxis_title="Time",
                yaxis_title="Health Score",
                showlegend=False
            )
            st.plotly_chart(fig_line, use_container_width=True)
        
        # Nutrition breakdown
        st.subheader("🥗 Nutrition Analysis")
        
        # Calculate average nutrition values from your dataset
        avg_nutrition = {
            'calories': sum([p['product_info']['calories'] for p in st.session_state.scanned_products]) / len(st.session_state.scanned_products),
            'proteins': sum([p['product_info']['protein_g'] for p in st.session_state.scanned_products]) / len(st.session_state.scanned_products),
            'carbs': sum([p['product_info']['carbs_g'] for p in st.session_state.scanned_products]) / len(st.session_state.scanned_products),
            'fat': sum([p['product_info']['fat_g'] for p in st.session_state.scanned_products]) / len(st.session_state.scanned_products)
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Calories", f"{avg_nutrition['calories']:.1f}")
            st.metric("Avg Proteins", f"{avg_nutrition['proteins']:.1f}g")
        
        with col2:
            st.metric("Avg Carbs", f"{avg_nutrition['carbs']:.1f}g")
            st.metric("Avg Fat", f"{avg_nutrition['fat']:.1f}g")
        
        with col3:
            # Calculate health score metrics
            avg_health_score = sum(health_scores) / len(health_scores)
            st.metric("Avg Health Score", f"{avg_health_score:.1f}%")
            
            high_score_products = len([s for s in health_scores if s >= 70])
            st.metric("Healthy Products", f"{high_score_products}/{len(health_scores)}")
        
        # Nutrition pie chart
        macros = ['Proteins', 'Carbs', 'Fat']
        macro_values = [avg_nutrition['proteins'], avg_nutrition['carbs'], avg_nutrition['fat']]
        
        if sum(macro_values) > 0:
            fig_pie = px.pie(
                values=macro_values,
                names=macros,
                title="Average Macronutrient Distribution",
                color_discrete_sequence=['#F18F01', '#C73E1D', '#2E86AB']
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Health warnings summary
        st.subheader("⚠️ Health Warnings Summary")
        
        all_warnings = []
        for product in st.session_state.scanned_products:
            all_warnings.extend(product['health_assessment']['warnings'])
        
        if all_warnings:
            warning_counts = Counter(all_warnings)
            
            # Display top warnings
            st.write("**Most Common Health Warnings:**")
            for warning, count in warning_counts.most_common(5):
                st.write(f"• {warning} ({count} times)")
        else:
            st.success("No health warnings for your scanned products! 🎉")
        
        # Recommendations summary
        st.subheader("💡 Recommendations Summary")
        
        all_recommendations = []
        for product in st.session_state.scanned_products:
            all_recommendations.extend(product['health_assessment']['recommendations'])
        
        if all_recommendations:
            rec_counts = Counter(all_recommendations)
            
            st.write("**Top Recommendations:**")
            for rec, count in rec_counts.most_common(5):
                st.write(f"• {rec} (mentioned {count} times)")
        else:
            st.info("Scan more products to get personalized recommendations!")
        
        # Category health analysis
        st.subheader("📊 Health by Category")
        
        category_health = {}
        for product in st.session_state.scanned_products:
            category = product['product_info']['category']
            health_score = product['health_score']
            
            if category not in category_health:
                category_health[category] = []
            category_health[category].append(health_score)
        
        # Calculate average health score by category
        if category_health:
            avg_category_health = {cat: sum(scores)/len(scores) for cat, scores in category_health.items()}
            
            categories = list(avg_category_health.keys())
            avg_scores = list(avg_category_health.values())
            
            fig_category = px.bar(
                x=categories,
                y=avg_scores,
                title="Average Health Score by Category",
                labels={'x': 'Category', 'y': 'Average Health Score'},
                color=avg_scores,
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_category, use_container_width=True)
    
    else:
        st.info("No products scanned yet. Use the Barcode Scanner to start tracking your health metrics!")
        
        # Show customer profile health tips
        if st.session_state.customer_profile['name']:
            st.subheader("🎯 Personalized Health Tips")
            
            conditions = st.session_state.customer_profile['conditions']
            age = st.session_state.customer_profile['age']
            activity_level = st.session_state.customer_profile['activity_level']
            
            if 'diabetes' in conditions:
                st.write("🩺 **For Diabetes Management:**")
                st.write("• Look for products with low carbohydrate content")
                st.write("• Choose high-fiber foods to help control blood sugar")
                st.write("• Avoid products with added sugars")
            
            if 'hypertension' in conditions:
                st.write("🫀 **For Blood Pressure Management:**")
                st.write("• Choose low-sodium products")
                st.write("• Include potassium-rich foods like fruits and vegetables")
                st.write("• Limit processed and packaged foods")
            
            if age > 65:
                st.write("👴 **For Seniors:**")
                st.write("• Ensure adequate protein intake for muscle health")
                st.write("• Choose calcium-rich foods for bone health")
                st.write("• Include vitamin D fortified products")
            
            if activity_level == 'high':
                st.write("🏃 **For Active Lifestyle:**")
                st.write("• Include high-protein foods for muscle recovery")
                st.write("• Choose complex carbohydrates for sustained energy")
                st.write("• Stay hydrated with low-sugar beverages")

# Tab 5: Notifications
with tab5:
    st.header("📧 Notifications & Alerts")
    
    # Email settings
    st.subheader("📧 Email Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        email_notifications = st.checkbox("Enable Email Notifications", value=True)
        health_alerts = st.checkbox("Health Score Alerts", value=True)
        shopping_reminders = st.checkbox("Shopping List Reminders", value=True)
    
    with col2:
        daily_summary = st.checkbox("Daily Summary", value=False)
        weekly_report = st.checkbox("Weekly Health Report", value=False)
        discount_alerts = st.checkbox("Discount Alerts", value=True)
    
    # Notification thresholds
    st.subheader("⚙️ Alert Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        health_threshold = st.slider("Health Score Alert Threshold", 0, 100, 40)
        st.markdown(f"Get alerts when products have health scores below {health_threshold}%")
    
    with col2:
        calorie_threshold = st.slider("High Calorie Alert", 0, 1000, 400)
        st.markdown(f"Get alerts when products have calories above {calorie_threshold} per 100g")
    
    # Recent notifications
    st.subheader("🔔 Recent Notifications")
    
    if st.session_state.notifications:
        # Display recent notifications
        for notification in reversed(st.session_state.notifications[-10:]):  # Show last 10
            if notification['type'] == 'success':
                st.success(f"**{notification['timestamp']}** - {notification['message']}")
            elif notification['type'] == 'warning':
                st.warning(f"**{notification['timestamp']}** - {notification['message']}")
            elif notification['type'] == 'error':
                st.error(f"**{notification['timestamp']}** - {notification['message']}")
            else:
                st.info(f"**{notification['timestamp']}** - {notification['message']}")
    else:
        st.info("No notifications yet. Start using the app to receive notifications!")
    
    # Test notification
    st.subheader("🧪 Test Notification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Send Test Email"):
            if st.session_state.customer_profile['email']:
                test_message = f"""
Hello {st.session_state.customer_profile['name']},

This is a test notification from your Smart Grocery Assistant.

Your current settings:
- Health Score Alert Threshold: {health_threshold}%
- Calorie Alert Threshold: {calorie_threshold} calories
- Email Notifications: {'Enabled' if email_notifications else 'Disabled'}

Recent Activity:
- Products Scanned: {len(st.session_state.scanned_products)}
- Shopping List Items: {len(st.session_state.shopping_list)}
- Total Weight Tracked: {st.session_state.total_weight:.2f}kg

Thank you for using Smart Grocery Assistant!

Best regards,
Your Smart Grocery Assistant Team
"""
                
                if send_email(
                    st.session_state.customer_profile['email'],
                    "Test Notification - Smart Grocery Assistant",
                    test_message
                ):
                    st.success("Test email sent successfully!")
                    add_notification("Test email sent successfully", "success")
                else:
                    st.error("Failed to send test email.")
            else:
                st.error("Please set your email in the profile settings.")
    
    with col2:
        if st.button("Clear All Notifications"):
            st.session_state.notifications = []
            st.success("All notifications cleared!")
            st.rerun()
    
    # Export data
    st.subheader("📤 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Scanned Products"):
            if st.session_state.scanned_products:
                # Create DataFrame
                export_data = []
                for product in st.session_state.scanned_products:
                    export_data.append({
                        'Product Name': product['product_info']['name'],
                        'Brand': product['product_info']['brand'],
                        'Category': product['product_info']['category'],
                        'Barcode': product['barcode'],
                        'Health Score': product['health_score'],
                        'Price': product['product_info']['price'],
                        'Calories': product['product_info']['calories'],
                        'Proteins': product['product_info']['protein_g'],
                        'Carbs': product['product_info']['carbs_g'],
                        'Fat': product['product_info']['fat_g'],
                        'Scan Time': product['scan_time']
                    })
                
                df = pd.DataFrame(export_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"scanned_products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No scanned products to export.")
    
    with col2:
        if st.button("Export Shopping List"):
            if st.session_state.shopping_list:
                # Create DataFrame
                df = pd.DataFrame(st.session_state.shopping_list)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"shopping_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No shopping list items to export.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🛒 Smart Grocery Assistant v2.0</p>
    <p>Powered by Your Custom Dataset | Built with Streamlit</p>
    <p>Making healthy shopping decisions easier, one scan at a time!</p>
</div>
""", unsafe_allow_html=True)