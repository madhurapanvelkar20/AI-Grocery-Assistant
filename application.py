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
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pytesseract
import easyocr
# If you're on Windows, you might need to specify the path:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'





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
            'Weight(grams)': 'weight_grams',
            'Barcode': 'barcode'  # Use your actual barcode column
        }
        
        # Rename columns to match expected format
        df = df.rename(columns=column_mapping)
        
        # Create missing required columns
        df['product_id'] = range(1, len(df) + 1)  # Generate product IDs
        df['weight_kg'] = df['weight_grams'] / 1000  # Convert grams to kg
        
        # Clean and validate data
        df = df.dropna(subset=['product_name', 'category', 'price'])
        df['health_tags'] = df['health_tags'].fillna('')
        df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce')
        df['weight_kg'] = df['weight_kg'].fillna(0.1)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['price'] = df['price'].fillna(0.0)
        df['calories'] = pd.to_numeric(df['calories'], errors='coerce')
        df['calories'] = df['calories'].fillna(0)
        df['protein_g'] = pd.to_numeric(df['protein_g'], errors='coerce')
        df['protein_g'] = df['protein_g'].fillna(0)
        df['fat_g'] = pd.to_numeric(df['fat_g'], errors='coerce')
        df['fat_g'] = df['fat_g'].fillna(0)
        df['carbs_g'] = pd.to_numeric(df['carbs_g'], errors='coerce')
        df['carbs_g'] = df['carbs_g'].fillna(0)
        df['popularity_score'] = pd.to_numeric(df['popularity_score'], errors='coerce')
        df['popularity_score'] = df['popularity_score'].fillna(0)
        
        # Clean barcodes - ensure they are strings and handle missing values
        df['barcode'] = df['barcode'].astype(str).str.strip()
        df['barcode'] = df['barcode'].replace(['nan', 'NaN', ''], pd.NA)
        
        # For products without barcodes, generate them
        missing_barcode_mask = df['barcode'].isna()
        if missing_barcode_mask.sum() > 0:
            df.loc[missing_barcode_mask, 'barcode'] = df.loc[missing_barcode_mask, 'product_id'].apply(
                lambda x: f"1234{str(x).zfill(9)}"
            )
        
        
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
        offers_df = pd.read_csv("product_offers.csv")
        
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
            
            
            return offers_df
        else:
            st.warning("Main dataset not loaded, cannot process offers")
            return create_sample_offers()
            
    except FileNotFoundError:
        st.info("product_offers (1).csv not found. Creating sample offers based on your dataset.")
        return create_sample_offers()
    except Exception as e:
        st.error(f"Error loading product_offers (1).csv: {str(e)}. Creating sample offers.")
        st.error(f"Make sure your offers file has columns: Product Name, Discounted Price (₹), Discount (%)")
        return create_sample_offers()

# === NEW  ➜  Load past orders and build USER×ITEM matrix ==============
@st.cache_data
def load_past_orders():
    """
    Returns
        user_item_matrix   (users × true numeric product_id)
        product_id_lookup  (list of numeric product_ids in same order)
    """
    try:
        past_orders_df = pd.read_csv("pastorders.csv").loc[:, ~pd.read_csv("pastorders.csv").columns.str.contains("^Unnamed")]

        # ── 1. add numeric product_id by joining with main catalogue ──────────
        main_df = load_data()                            # your big products CSV
        past_orders_df = past_orders_df.merge(main_df[['product_id', 'product_name']],
                      on='product_name', how='left')

        # drop rows that could not be matched
        past_orders_df = past_orders_df.dropna(subset=['product_id'])

        # ── 2. normalise column names ─────────────────────────────────────────
        past_orders_df = past_orders_df.rename(columns={
            "customer_email": "user_id",
            # keep *numeric* product_id we just merged in
            "rating": "rating"          # or use 'quantity'
        })

        if "rating" not in past_orders_df.columns:
            past_orders_df["rating"] = 1            # implicit feedback

        # ── 3. pivot USER × ITEM ──────────────────────────────────────────────
        user_item_matrix = (
            past_orders_df.pivot_table(index="user_id",
                           columns="product_id",         # numeric!
                           values="rating",
                           aggfunc="sum",
                           fill_value=0)
        )

        return user_item_matrix, list(user_item_matrix.columns)

    except Exception as e:
        st.warning(f"Could not load pastorders.csv → {e}")
        return pd.DataFrame(), []


def get_user_item_matrix():
       """Convert shopping list data into user-item matrix"""
       if not st.session_state.shopping_list:
           return None
       
       interactions = pd.DataFrame({
           'user_id': [st.session_state.customer_profile['email']]*len(st.session_state.shopping_list),
           'product_id': [item['product_id'] for item in st.session_state.shopping_list],
           'interaction': [1]*len(st.session_state.shopping_list)
       })
       
       return interactions.pivot_table(
           index='user_id',
           columns='product_id',
           values='interaction',
           fill_value=0
       )

# === NEW  ➜  Global similarity matrix (computed once, cached) =========
@st.cache_data
def get_item_similarity():
    """Return (item_similarity_matrix, product_id_list) ready for use."""
    uim, product_ids = load_past_orders()   # call the helper we just wrote
    if uim.empty or uim.shape[0] < 2:
        return None, []
    sim = cosine_similarity(uim.T)          # item‑based similarity
    return sim, product_ids

def get_recommendations(product_id, item_similarity, product_ids, n=5):
    """Get top-n similar products to given product_id"""
    if product_id not in product_ids:
        return []

    idx = product_ids.index(product_id)
    similar_items = sorted(enumerate(item_similarity[idx]),
                           key=lambda x: x[1], reverse=True)[1:n+1]

    return [product_ids[i] for i, _ in similar_items]


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

# --- Enhanced Barcode Detection with OCR ---
def detect_barcode_region(image):
    """Automatically detect and read barcode, then find product - SINGLE SCAN ONLY"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        st.info("🔍 Analyzing image for barcode...")
        
        # Try different methods but STOP after first success
        extracted_barcode = None
        processed_image = gray
        
        # Method 1: Direct OCR on entire image
        extracted_barcode = extract_barcode_with_easyocr(gray)
        if extracted_barcode:
            return auto_process_barcode(extracted_barcode, gray)
        
        # Method 2: Detect region first, then OCR
        detected_regions = detect_multiple_barcode_regions(gray)
        
        for region_info in detected_regions:
            if region_info['detected']:
                x, y, w, h = region_info['region']
                roi = gray[y:y+h, x:x+w]
                
                # Try OCR on detected region
                extracted_barcode = extract_barcode_with_easyocr(roi)
                if extracted_barcode:
                    return auto_process_barcode(extracted_barcode, region_info['processed_image'])
        
        # Method 3: Try preprocessing and OCR
        preprocessed_images = preprocess_for_ocr(gray)
        for proc_img in preprocessed_images:
            extracted_barcode = extract_barcode_with_easyocr(proc_img)
            if extracted_barcode:
                return auto_process_barcode(extracted_barcode, proc_img)
        
        return {
            'detected': False, 
            'region': None, 
            'confidence': 0, 
            'processed_image': gray, 
            'extracted_barcode': None,
            'product_info': None,
            'error_message': 'Could not extract barcode from image'
        }
    
    except Exception as e:
        return {
            'detected': False, 
            'region': None, 
            'confidence': 0, 
            'processed_image': None, 
            'extracted_barcode': None,
            'product_info': None,
            'error_message': f'Error in detection: {str(e)}'
        }

def extract_barcode_with_easyocr(image):
    """Hybrid approach: Try fast methods first, then EasyOCR"""
    
    # Method 1: Try Pytesseract first (fastest)
    try:
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(image, config=custom_config).strip()
        cleaned_text = ''.join(filter(str.isdigit, text))
        if len(cleaned_text) >= 8 and len(cleaned_text) <= 13:
            return cleaned_text
    except:
        pass
    
    # Method 2: Try simple pattern matching on image
    try:
        # Look for barcode-like patterns without OCR
        text_regions = find_text_regions(image)
        for region in text_regions:
            # Simple digit extraction
            if has_barcode_pattern(region):
                return extract_digits_simple(region)
    except:
        pass
    
    # Method 3: EasyOCR as last resort (slowest but most accurate)
    try:
        reader = get_easyocr_reader()
        results = reader.readtext(image, width_ths=0.7, height_ths=0.7)
        
        for (bbox, text, confidence) in results:
            cleaned_text = ''.join(filter(str.isdigit, text))
            if len(cleaned_text) >= 8 and len(cleaned_text) <= 13:
                return cleaned_text
    except:
        pass
    
    return None

def find_text_regions(image):
    """Find potential text regions quickly"""
    # Simple contour-based text detection
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 10 and w/h > 3:  # Barcode-like dimensions
            text_regions.append(binary[y:y+h, x:x+w])
    
    return text_regions

def has_barcode_pattern(region):
    """Quick check if region looks like barcode text"""
    # Simple heuristics
    height, width = region.shape
    return width > height * 3 and width > 50

def extract_digits_simple(region):
    """Simple digit extraction without heavy OCR"""
    # Template matching or simple pattern recognition
    # This is a placeholder - implement basic digit recognition
    return None

def preprocess_for_ocr(image):
    """Create multiple preprocessed versions for better OCR"""
    preprocessed = []
    
    try:
        # Original image
        preprocessed.append(image)
        
        # Resize image (OCR works better on larger images)
        height, width = image.shape
        if height < 200:
            scale = 200 / height
            new_width = int(width * scale)
            resized = cv2.resize(image, (new_width, 200))
            preprocessed.append(resized)
        
        # Binary threshold
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed.append(binary)
        
        # Inverted binary
        _, inv_binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        preprocessed.append(inv_binary)
        
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        preprocessed.append(adaptive)
        
        # Gaussian blur + threshold
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, blur_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed.append(blur_thresh)
        
        return preprocessed
    except:
        return [image]

def detect_multiple_barcode_regions(gray):
    """Detect potential barcode regions using multiple methods"""
    regions = []
    
    # Method 1: Gradient-based
    try:
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        
        blurred_grad = cv2.blur(gradient, (9, 9))
        (_, thresh) = cv2.threshold(blurred_grad, 225, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            x, y, w, h = cv2.boundingRect(contour)
            if w > h and w > 50 and h > 10:
                regions.append({
                    'detected': True,
                    'region': (x, y, w, h),
                    'confidence': cv2.contourArea(contour),
                    'processed_image': closed
                })
    except:
        pass
    
    # Method 2: Simple threshold
    try:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 40 and h > 8 and w/h > 1.5:
                regions.append({
                    'detected': True,
                    'region': (x, y, w, h),
                    'confidence': cv2.contourArea(contour),
                    'processed_image': binary
                })
    except:
        pass
    
    return regions if regions else [{'detected': False, 'region': None, 'confidence': 0, 'processed_image': None}]

def auto_process_barcode(barcode, processed_image):
    """Automatically process detected barcode and find product - SINGLE SCAN ONLY"""
    try:
        st.success(f"🎉 Barcode detected: {barcode}")
        
        # Validate barcode
        if not validate_barcode(barcode):
            return {
                'detected': True,
                'region': None,
                'confidence': 100,
                'processed_image': processed_image,
                'extracted_barcode': barcode,
                'product_info': None,
                'error_message': 'Invalid barcode format'
            }
        
        # Search for product
        with st.spinner("🔍 Searching for product..."):
            product_info = simulate_barcode_scan(barcode)
        
        if product_info['found']:
            # Product found - assess health
            health_assessment = assess_health_for_customer(product_info, st.session_state.customer_profile)
            
            # CHECK: Only add if not already scanned
            existing_scan = None
            for scan in st.session_state.scanned_products:
                if scan['barcode'] == barcode:
                    existing_scan = scan
                    break
            
            if not existing_scan:
                # Add to scanned products ONLY if not already there
                scan_result = {
                    'barcode': barcode,
                    'product_info': product_info,
                    'health_assessment': health_assessment,
                    'health_score': health_assessment['score'],
                    'scan_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.session_state.scanned_products.append(scan_result)
                add_notification(f"Auto-scanned {product_info['name']}", "success")
           
            
            return {
                'detected': True,
                'region': None,
                'confidence': 100,
                'processed_image': processed_image,
                'extracted_barcode': barcode,
                'product_info': product_info,
                'health_assessment': health_assessment,
                'success': True
            }
        else:
            return {
                'detected': True,
                'region': None,
                'confidence': 100,
                'processed_image': processed_image,
                'extracted_barcode': barcode,
                'product_info': None,
                'error_message': 'Product not found in database'
            }
    
    except Exception as e:
        return {
            'detected': True,
            'region': None,
            'confidence': 100,
            'processed_image': processed_image,
            'extracted_barcode': barcode,
            'product_info': None,
            'error_message': f'Error processing barcode: {str(e)}'
        }
        
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
            'fat_g': product_info.get('fat_g', 0),
            'carbs_g': product_info.get('carbs_g', 0),
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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Home Dashboard", "🔍 Barcode Scanner", "📋 My Products", "📊 Health Dashboard", "📧 Notifications"])

# Tab 1: Home Dashboard
with tab1:
    st.header("🏠 Home Dashboard")
    
    # Welcome message
    if st.session_state.customer_profile['name']:
        st.markdown(f"Welcome back, **{st.session_state.customer_profile['name']}**! 👋")
    else:
        st.markdown("Welcome to Smart Grocery Assistant! Please set up your profile in the sidebar.")
    
    # Image Carousel
    st.subheader("🎯 Your Smart Shopping Journey")
    
    # Create carousel with 4 images
    carousel_images = [
        {
            "url": "https://images.unsplash.com/photo-1542838132-92c53300491e?w=800&h=400&fit=crop",
            "caption": "🛒 Browse Products",
            "description": "Discover thousands of products with detailed nutritional information"
        },
        {
            "url": "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=800&h=400&fit=crop",
            "caption": "📱 Scan Barcodes", 
            "description": "Quick barcode scanning with instant health insights"
        },
        {
            "url": "https://images.unsplash.com/photo-1498837167922-ddd27525d352?w=800&h=400&fit=crop",
            "caption": "🥗 Health Analysis",
            "description": "Get personalized health recommendations for every product"
        },
        {
            "url": "https://images.unsplash.com/photo-1601599561213-832382fd07ba?w=800&h=400&fit=crop",
            "caption": "✅ Smart Shopping",
            "description": "Build your perfect shopping list with AI recommendations"
        }
    ]
    
    # Display carousel using columns
    cols = st.columns(4)
    
    for i, (col, image_data) in enumerate(zip(cols, carousel_images)):
        with col:
            st.image(image_data["url"], caption=image_data["caption"], use_container_width=True)
            st.markdown(f"**{image_data['description']}**")
    
    st.markdown("---")
    
    # Quick stats
    col2, col3, col4 = st.columns(3)
    
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
        # Check if there are any offers available
        if not offers_df.empty:
            st.markdown(
                "<div style='text-align: center; margin: 20px 0;'>"
                "<h3 style='color: #000000; font-weight: bold;'>"
                "🎉 EXCITING OFFERS AVAILABLE 🔥"
                "</h3></div>", 
                unsafe_allow_html=True
            )
        
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
            filtered_df = df[df['category'] == selected_category].copy()
        
        if selected_health_filter != "All":
            filter_map = {
                "High Protein": "protein",
                "High Fiber": "fiber",
                "Organic": "organic",
                "Low Calorie": "low calorie"
            }
            filter_tag = filter_map.get(selected_health_filter, "")
            if filter_tag:
                filtered_df = filtered_df[filtered_df['health_tags'].str.contains(filter_tag, case=False, na=False)].copy()
        
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
                            offer = offers_df[offers_df['product_id'] == product['product_id']].copy()
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
       
    # ===== Add This Recommendation Section =====
    if st.session_state.shopping_list:
        st.subheader("🤖 Recommended For You")
        
        # Get recommendations based on current shopping list
        item_similarity, product_ids = get_item_similarity()
        
        if item_similarity is not None and product_ids:
            # Get recent products from shopping list
            recent_products = [item['product_id'] for item in st.session_state.shopping_list[-3:]]
            
            # Collect all recommendations
            all_recommendations = []
            for pid in recent_products:
                if pid in product_ids:
                    recs = get_recommendations(pid, item_similarity, product_ids, n=3)
                    all_recommendations.extend(recs)
            
            # Remove duplicates and limit to 6 recommendations
            unique_recommendations = list(set(all_recommendations))[:6]
            
            if unique_recommendations:
                st.write("Based on your shopping list, you might also like:")
                
                # Display recommendations in a grid
                cols = st.columns(3)
                for i, rec_pid in enumerate(unique_recommendations):
                    if i < 6:  # Limit to 6 recommendations
                        with cols[i % 3]:
                            # Find product in main dataset
                            product = df[df['product_id'] == rec_pid]
                            if not product.empty:
                                product = product.iloc[0]
                                
                                # Create a card-like display
                                st.markdown(f"**{product['product_name']}**")
                                if 'brand' in product and pd.notna(product['brand']):
                                    st.write(f"*{product['brand']}*")
                                st.write(f"**₹{product['price']:.2f}**")
                                st.write(f"Category: {product['category']}")
                                
                                # Show health tags if available
                                if pd.notna(product['health_tags']):
                                    tags = product['health_tags'].split(',')[:2]
                                    for tag in tags:
                                        st.write(f"• {tag.strip()}")
                                
                                # Add to shopping list button
                                if st.button(f"Add to Cart", key=f"rec_add_{rec_pid}"):
                                    product_info = {
                                        'found': True,
                                        'product_id': rec_pid,
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
                st.info("Add more items to your shopping list to get personalized recommendations!")
        else:
            st.info("No recommendation data available. Add items to your shopping list to get started!")

# Tab 2: Barcode Scanner
with tab2:
    st.header("🔍 Barcode Scanner")
    
    # Option 1: Upload image
    st.subheader("📷 Upload Barcode Image")
    uploaded_file = st.file_uploader("Choose an image with barcode", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Detect barcode region
        with st.spinner("Detecting barcode region..."):
            detection_result = detect_barcode_region(image)
            
            if detection_result['detected']:
                st.success(f"Barcode region detected! ")
                
                # Show processed image
                if detection_result['processed_image'] is not None:
                    st.image(detection_result['processed_image'], caption="Processed Image", use_container_width=True)
                
                # Check if barcode was automatically extracted
                if detection_result.get('extracted_barcode'):
                    st.success(f"🎉 Automatically detected barcode: {detection_result['extracted_barcode']}")
                    barcode_input = detection_result['extracted_barcode']
                    
                    # Auto-scan the detected barcode
                    if st.button("🚀 Auto-Scan Detected Barcode", type="primary"):
                        if validate_barcode(barcode_input):
                            with st.spinner("Finding product..."):
                                product_info = simulate_barcode_scan(barcode_input)
                                
                                if product_info['found']:
                                    # Assess health and add to scanned products
                                    health_assessment = assess_health_for_customer(product_info, st.session_state.customer_profile)
                                    scan_result = {
                                        'barcode': barcode_input,
                                        'product_info': product_info,
                                        'health_assessment': health_assessment,
                                        'health_score': health_assessment['score'],
                                        'scan_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    }
                                    st.session_state.scanned_products.append(scan_result)
                                    st.success("Product automatically scanned!")
                                    add_notification(f"Auto-scanned {product_info['name']}", "success")
                                    st.rerun()
                                else:
                                    st.error("Product not found in database.")
                        else:
                            st.error("Invalid barcode format.")
                else:
                    st.info("Please enter the barcode numbers from the detected region above:")
                    detected_barcode = st.text_input("Detected barcode:", key="detected_barcode")
                    if detected_barcode:
                        barcode_input = detected_barcode
            else:
                st.warning("No barcode region detected. Please try a clearer image or enter manually.")
    
    # Option 2: Manual barcode entry
    st.subheader("⌨️ Manual Barcode Entry")
    manual_barcode = st.text_input("Enter barcode number:", key="manual_barcode")
    
    if manual_barcode and st.button("🔍 Scan Barcode"):
        if validate_barcode(manual_barcode):
            with st.spinner("Finding product..."):
                product_info = simulate_barcode_scan(manual_barcode)
                
                if product_info['found']:
                    # Display product information
                    st.success("✅ Product Found!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(product_info['name'])
                        st.write(f"**Brand:** {product_info['brand']}")
                        st.write(f"**Category:** {product_info['category']}")
                        st.write(f"**Price:** ₹{product_info['price']:.2f}")
                        st.write(f"**Weight:** {product_info['weight_kg']:.3f} kg")
                        
                        if product_info['ingredients']:
                            st.write(f"**Ingredients:** {product_info['ingredients'][:100]}...")
                    
                    with col2:
                        st.write("**Nutrition (per 100g):**")
                        st.write(f"• Calories: {product_info['calories']}")
                        st.write(f"• Protein: {product_info['protein_g']}g")
                        st.write(f"• Carbs: {product_info['carbs_g']}g")
                        st.write(f"• Fat: {product_info['fat_g']}g")
                        
                        if product_info['health_tags']:
                            st.write("**Health Tags:**")
                            for tag in product_info['health_tags'][:5]:
                                st.write(f"• {tag}")
                    
                    # Health assessment
                    health_assessment = assess_health_for_customer(product_info, st.session_state.customer_profile)
                    
                    # Display health score with color coding
                    score = health_assessment['score']
                    if score >= 70:
                        st.success(f"🟢 Health Score: {score}% - Excellent choice!")
                    elif score >= 40:
                        st.warning(f"🟡 Health Score: {score}% - Moderate choice")
                    else:
                        st.error(f"🔴 Health Score: {score}% - Consider alternatives")
                    
                    # Display warnings and recommendations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if health_assessment['warnings']:
                            st.subheader("⚠️ Health Warnings")
                            for warning in health_assessment['warnings']:
                                st.write(f"• {warning}")
                    
                    with col2:
                        if health_assessment['recommendations']:
                            st.subheader("💡 Recommendations")
                            for rec in health_assessment['recommendations']:
                                st.write(f"• {rec}")
                    
                    # Add to lists buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        quantity = st.number_input("Quantity:", min_value=1, value=1, key="scan_quantity")
                        if st.button("🛒 Add to Shopping List"):
                            if add_to_shopping_list(product_info, quantity):
                                st.success(f"Added {quantity}x {product_info['name']} to shopping list!")
                                add_notification(f"Added {product_info['name']} to shopping list", "success")
                                st.rerun()
                    
                    with col2:
                        st.write("")  # Spacing
                        if st.button("📝 Save to Scanned Products"):
                            scan_result = {
                                'barcode': manual_barcode,
                                'product_info': product_info,
                                'health_assessment': health_assessment,
                                'health_score': health_assessment['score'],
                                'scan_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            st.session_state.scanned_products.append(scan_result)
                            st.success("Product saved to scanned products!")
                            add_notification(f"Scanned {product_info['name']}", "success")
                            st.rerun()
                
                else:
                    st.error("❌ Product not found in database.")
                    st.info("Please check the barcode number or try a different product.")
        else:
            st.error("❌ Invalid barcode format. Please enter a valid barcode (8-13 digits).")
    
    # Recent scans
    if st.session_state.scanned_products:
        st.subheader("📋 Recent Scans")
        for i, scan in enumerate(reversed(st.session_state.scanned_products[-5:])):  # Show last 5
            with st.expander(f"{scan['product_info']['name']} - {scan['scan_time']}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Barcode:** {scan['barcode']}")
                    st.write(f"**Price:** ₹{scan['product_info']['price']:.2f}")
                    st.write(f"**Category:** {scan['product_info']['category']}")
                
                with col2:
                    score = scan['health_score']
                    score_color = "🟢" if score >= 70 else "🟡" if score >= 40 else "🔴"
                    st.write(f"**Health Score:** {score_color} {score}%")
                    
                    if st.button(f"Add to Cart", key=f"recent_scan_{i}"):
                        if add_to_shopping_list(scan['product_info']):
                            st.success(f"Added {scan['product_info']['name']} to cart!")
                            st.rerun()

# Tab 3: Unified Shopping List & Scanned Products
with tab3:
    st.header("📋 My Products ")
    
    # Add manual item section
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
                filtered_products = df.copy()
            else:
                filtered_products = df[df['category'] == selected_category].copy()
            
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
    else:
        st.error("No products loaded from dataset. Please check your dataset.csv file.")
    
    # Create unified product view
    all_products = []
    
    # Add shopping list items
    for idx, item in enumerate(st.session_state.shopping_list):
        all_products.append({
            'type': 'shopping_list',
            'index': idx,
            'product_id': item['product_id'],
            'name': item['name'],
            'brand': item.get('brand', 'Unknown'),
            'category': item['category'],
            'price': item['price'],
            'quantity': item['quantity'],
            'total_price': item.get('total_price', 0),
            'total_weight': item.get('total_weight', 0),
            'weight_kg': item['weight_kg'],
            'health_tags': item.get('health_tags', []),
            'calories': item.get('calories', 0),
            'protein_g': item.get('protein_g', 0),
            'fat_g': item.get('fat_g', 0),
            'carbs_g': item.get('carbs_g', 0),
            'added_time': item.get('added_time', ''),
            'health_score': None,  # Calculate if needed
            'status': 'In Cart'
        })
    
    # Add scanned products that are NOT in shopping list
    shopping_list_product_ids = [item['product_id'] for item in st.session_state.shopping_list]
    
    for idx, scan in enumerate(st.session_state.scanned_products):
        if scan['product_info']['product_id'] not in shopping_list_product_ids:
            all_products.append({
                'type': 'scanned',
                'index': idx,
                'product_id': scan['product_info']['product_id'],
                'name': scan['product_info']['name'],
                'brand': scan['product_info']['brand'],
                'category': scan['product_info']['category'],
                'price': scan['product_info']['price'],
                'quantity': 0,  # Not in cart yet
                'total_price': 0,
                'total_weight': 0,
                'weight_kg': scan['product_info']['weight_kg'],
                'health_tags': scan['product_info'].get('health_tags', []),
                'calories': scan['product_info'].get('calories', 0),
                'protein_g': scan['product_info'].get('protein_g', 0),
                'fat_g': scan['product_info'].get('fat_g', 0),
                'carbs_g': scan['product_info'].get('carbs_g', 0),
                'added_time': scan['scan_time'],
                'health_score': scan['health_score'],
                'status': 'Scanned',
                'barcode': scan['barcode'],
                'health_assessment': scan['health_assessment']
            })
    
    # Display unified product list
    if all_products:
        st.subheader("📦 All Your Products")
        
        # Summary statistics
        shopping_items = [p for p in all_products if p['type'] == 'shopping_list']
        scanned_items = [p for p in all_products if p['type'] == 'scanned']
        
        total_items_in_cart = sum([item['quantity'] for item in shopping_items])
        total_price = sum([item['total_price'] for item in shopping_items])
        total_weight_kg = sum([item['total_weight'] for item in shopping_items])
        
        col1, col3, col4,  = st.columns(3)
        with col1:
            st.metric("In Cart", total_items_in_cart)
        
        with col3:
            st.metric("Total Price", f"₹{total_price:.2f}")
        with col4:
            st.metric("Total Weight", f"{total_weight_kg:.2f} kg")
        
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            status_filter = st.selectbox("Filter by Status", ["All", "In Cart", "Scanned Only"])
        with col2:
            category_filter = st.selectbox("Filter by Category", ["All"] + sorted(list(set([p['category'] for p in all_products]))))
        
        # Apply filters
        filtered_products = all_products.copy()
        if status_filter != "All":
            if status_filter == "In Cart":
                filtered_products = [p for p in filtered_products if p['type'] == 'shopping_list']
            elif status_filter == "Scanned Only":
                filtered_products = [p for p in filtered_products if p['type'] == 'scanned']
        
        if category_filter != "All":
            filtered_products = [p for p in filtered_products if p['category'] == category_filter]
        
        # Sort products: Shopping list items first, then by time
        filtered_products.sort(key=lambda x: (x['type'] != 'shopping_list', x['added_time']))
        
        # Display products
        for product in filtered_products:
            with st.container():
                # Different styling based on status
                if product['type'] == 'shopping_list':
                    st.markdown("🛒 **IN CART**")
                else:
                    st.markdown("👁️ **SCANNED**")
                
                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 2])
                
                with col1:
                    st.markdown(f"**{product['name']}**")
                    if product['brand'] != 'Unknown':
                        st.markdown(f"*{product['brand']}*")
                    st.markdown(f"**Category:** {product['category']}")
                    st.markdown(f"**Price:** ₹{product['price']:.2f}")
                    
                    # Show health score if available
                    if product.get('health_score') is not None:
                        score_color = "🟢" if product['health_score'] >= 70 else "🟡" if product['health_score'] >= 40 else "🔴"
                        st.markdown(f"**Health Score:** {score_color} {product['health_score']:.1f}%")
                    
                    # Show nutritional info
                    if product['calories'] > 0:
                        st.markdown(f"**Nutrition:** {product['calories']} cal | {product['protein_g']}g protein | {product['carbs_g']}g carbs")
                    
                    # Show health tags
                    if product['health_tags']:
                        tags_display = ", ".join(product['health_tags'][:3])
                        st.markdown(f"**Tags:** {tags_display}")
                
                with col2:
                    if product['type'] == 'shopping_list':
                        # Quantity control for shopping list items
                        new_quantity = st.number_input(
                            "Qty", 
                            min_value=1, 
                            value=product['quantity'], 
                            key=f"qty_unified_{product['product_id']}_{product['index']}"
                        )
                        if new_quantity != product['quantity']:
                            # Update shopping list
                            old_total_price = st.session_state.shopping_list[product['index']]['total_price']
                            old_total_weight = st.session_state.shopping_list[product['index']]['total_weight']
                            
                            st.session_state.shopping_list[product['index']]['quantity'] = new_quantity
                            st.session_state.shopping_list[product['index']]['total_price'] = product['price'] * new_quantity
                            st.session_state.shopping_list[product['index']]['total_weight'] = product['weight_kg'] * new_quantity
                            
                            # Update session totals
                            st.session_state.total_cost = st.session_state.total_cost - old_total_price + st.session_state.shopping_list[product['index']]['total_price']
                            st.session_state.total_weight = st.session_state.total_weight - old_total_weight + st.session_state.shopping_list[product['index']]['total_weight']
                            
                            st.rerun()
                    else:
                        # Add to cart option for scanned items
                        add_quantity = st.number_input(
                            "Add Qty", 
                            min_value=1, 
                            value=1, 
                            key=f"add_qty_{product['product_id']}_{product['index']}"
                        )
                
                with col3:
                    if product['type'] == 'shopping_list':
                        st.markdown(f"**Total:**")
                        st.markdown(f"₹{product['total_price']:.2f}")
                    else:
                        st.markdown(f"**Price:**")
                        st.markdown(f"₹{product['price']:.2f}")
                
                with col4:
                    if product['type'] == 'shopping_list':
                        st.markdown(f"**Weight:**")
                        st.markdown(f"{product['total_weight']:.3f} kg")
                    else:
                        st.markdown(f"**Weight:**")
                        st.markdown(f"{product['weight_kg']:.3f} kg")
                
                with col5:
                    if product['type'] == 'shopping_list':
                        # Remove from cart button
                        if st.button("🗑️ Remove", key=f"remove_{product['product_id']}_{product['index']}", help="Remove from cart"):
                            remove_from_shopping_list(product['index'])
                            add_notification(f"Removed {product['name']} from shopping list", "info")
                            st.rerun()
                    else:
                        # Add to cart button for scanned items
                        if st.button("🛒 Add to Cart", key=f"add_to_cart_{product['product_id']}_{product['index']}"):
                            # Create product info for adding to shopping list
                            product_info = {
                                'found': True,
                                'product_id': product['product_id'],
                                'name': product['name'],
                                'brand': product['brand'],
                                'category': product['category'],
                                'price': product['price'],
                                'weight_kg': product['weight_kg'],
                                'health_tags': product['health_tags'],
                                'calories': product['calories'],
                                'protein_g': product['protein_g'],
                                'fat_g': product['fat_g'],
                                'carbs_g': product['carbs_g']
                            }
                            
                            if add_to_shopping_list(product_info, add_quantity):
                                st.success(f"Added {product['name']} to cart!")
                                add_notification(f"Added {product['name']} to shopping list", "success")
                                st.rerun()
                
                # Show health warnings/recommendations for scanned items
                if product['type'] == 'scanned' and product.get('health_assessment'):
                    with st.expander("🔍 Health Details", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if product['health_assessment']['warnings']:
                                st.markdown("**⚠️ Warnings:**")
                                for warning in product['health_assessment']['warnings']:
                                    st.markdown(f"• {warning}")
                        
                        with col2:
                            if product['health_assessment']['recommendations']:
                                st.markdown("**💡 Recommendations:**")
                                for rec in product['health_assessment']['recommendations']:
                                    st.markdown(f"• {rec}")
                
                # Show barcode for scanned items
                if product['type'] == 'scanned' and product.get('barcode'):
                    st.markdown(f"**🔖 Barcode:** {product['barcode']}")
                
                st.markdown("---")
        
        # Action buttons for shopping list items
        if shopping_items:
            st.markdown("### 🛒 Shopping Cart Actions")
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
                if st.button("📧 Email Cart"):
                    if st.session_state.customer_profile['email']:
                        # Create email content
                        email_content = f"""
Shopping Cart for {st.session_state.customer_profile['name']}
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Items in Cart:
"""
                        for item in st.session_state.shopping_list:
                            email_content += f"• {item['name']} - Qty: {item['quantity']} - ₹{item.get('total_price', 0):.2f}\n"
                        
                        email_content += f"""
Total Items: {total_items_in_cart}
Total Price: ₹{total_price:.2f}

Generated by Smart Grocery Assistant
"""
                        
                        if send_email(
                            st.session_state.customer_profile['email'],
                            "Your Shopping Cart",
                            email_content
                        ):
                            st.success("Shopping cart sent to your email!")
                            add_notification("Shopping cart emailed successfully", "success")
                        else:
                            st.error("Failed to send email. Please check your email settings.")
                    else:
                        st.error("Please set your email in the profile settings.")
            
            with col3:
                if st.button("🗑️ Clear Cart"):
                    st.session_state.shopping_list = []
                    st.session_state.total_cost = 0
                    st.session_state.total_weight = 0
                    add_notification("Cleared all items from shopping cart", "info")
                    st.rerun()
        
        # Analytics section
        if all_products:
            st.markdown("---")
            st.subheader("📊 Product Analytics")
            
            # Create category summary
            category_summary = {}
            for product in all_products:
                category = product['category']
                if category not in category_summary:
                    category_summary[category] = {
                        'total_products': 0,
                        'in_cart': 0,
                        'scanned_only': 0,
                        'total_cost': 0,
                        'total_weight': 0
                    }
                
                category_summary[category]['total_products'] += 1
                
                if product['type'] == 'shopping_list':
                    category_summary[category]['in_cart'] += product['quantity']
                    category_summary[category]['total_cost'] += product['total_price']
                    category_summary[category]['total_weight'] += product['total_weight']
                else:
                    category_summary[category]['scanned_only'] += 1
            
            # Display category breakdown
            for category, data in category_summary.items():
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.write(f"**{category}**")
                with col2:
                    st.write(f"Total: {data['total_products']}")
                with col3:
                    st.write(f"In Cart: {data['in_cart']}")
                with col4:
                    st.write(f"Cost: ₹{data['total_cost']:.2f}")
                with col5:
                    st.write(f"Weight: {data['total_weight']:.2f} kg")
            
            # Visualizations
            if len(category_summary) > 1:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Cost distribution pie chart
                    categories = list(category_summary.keys())
                    costs = [category_summary[cat]['total_cost'] for cat in categories]
                    
                    if sum(costs) > 0:
                        fig_pie = px.pie(
                            values=costs,
                            names=categories,
                            title="Cart Value by Category"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Product distribution bar chart
                    in_cart = [category_summary[cat]['in_cart'] for cat in categories]
                    scanned = [category_summary[cat]['scanned_only'] for cat in categories]
                    
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(name='In Cart', x=categories, y=in_cart))
                    fig_bar.add_trace(go.Bar(name='Scanned Only', x=categories, y=scanned))
                    fig_bar.update_layout(
                        title="Products by Category & Status",
                        barmode='stack',
                        xaxis_title="Category",
                        yaxis_title="Number of Products"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
    
    else:
        st.info("🛒 No products yet!")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Start by:**")
            st.write("• 📱 Scanning barcodes in the Scanner tab")
            st.write("• 🏠 Browsing products in the Home Dashboard") 
            st.write("• ➕ Adding items from the catalog above")
        with col2:
            st.write("**Benefits:**")
            st.write("• 🎯 Get health insights for every product")
            st.write("• 💡 Receive personalized recommendations")
            st.write("• 📊 Track your shopping patterns")

# Tab 4: Health Dashboard
with tab4:
    st.header("📊 Health Dashboard")
    
    # Collect ALL products with health data (scanned + shopping list items)
    all_health_products = []
    
    # Add scanned products
    for product in st.session_state.scanned_products:
        all_health_products.append({
            'name': product['product_info']['name'],
            'category': product['product_info']['category'],
            'health_score': product['health_score'],
            'calories': product['product_info']['calories'],
            'protein_g': product['product_info']['protein_g'],
            'carbs_g': product['product_info']['carbs_g'],
            'fat_g': product['product_info']['fat_g'],
            'scan_time': product['scan_time'],
            'source': 'Scanned',
            'health_assessment': product['health_assessment']
        })
    
    # Add shopping list items (assess their health if not already done)
    for item in st.session_state.shopping_list:
        # Create product_info for health assessment
        product_info = {
            'found': True,
            'name': item['name'],
            'category': item['category'],
            'calories': item.get('calories', 0),
            'protein_g': item.get('protein_g', 0),
            'fat_g': item.get('fat_g', 0),
            'carbs_g': item.get('carbs_g', 0),
            'health_tags': item.get('health_tags', [])
        }
        
        # Assess health for shopping list item
        health_assessment = assess_health_for_customer(product_info, st.session_state.customer_profile)
        
        all_health_products.append({
            'name': item['name'],
            'category': item['category'],
            'health_score': health_assessment['score'],
            'calories': item.get('calories', 0),
            'protein_g': item.get('protein_g', 0),
            'carbs_g': item.get('carbs_g', 0),
            'fat_g': item.get('fat_g', 0),
            'scan_time': item.get('added_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            'source': 'Added to Cart',
            'health_assessment': health_assessment
        })
    
    if all_health_products:
        st.info(f"📊 Analyzing Products")
        
        # Health score distribution
        health_scores = [p['health_score'] for p in all_health_products]
        
        # Nutrition breakdown
        st.subheader("🥗 Nutrition Analysis (All Products)")
        
        # Calculate average nutrition values
        avg_nutrition = {
            'calories': sum([p['calories'] for p in all_health_products]) / len(all_health_products),
            'proteins': sum([p['protein_g'] for p in all_health_products]) / len(all_health_products),
            'carbs': sum([p['carbs_g'] for p in all_health_products]) / len(all_health_products),
            'fat': sum([p['fat_g'] for p in all_health_products]) / len(all_health_products)
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
                title="Average Macronutrient Distribution (All Products)",
                color_discrete_sequence=['#F18F01', '#C73E1D', '#2E86AB']
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Health warnings summary
        st.subheader("⚠️ Health Warnings Summary")
        
        all_warnings = []
        for product in all_health_products:
            all_warnings.extend(product['health_assessment']['warnings'])
        
        if all_warnings:
            warning_counts = Counter(all_warnings)
            
            # Display top warnings
            st.write("**Most Common Health Warnings:**")
            for warning, count in warning_counts.most_common(5):
                st.write(f"• {warning} ({count} times)")
        else:
            st.success("No health warnings for your products! 🎉")
        
        # Recommendations summary
        st.subheader("💡 Recommendations Summary")
        
        all_recommendations = []
        for product in all_health_products:
            all_recommendations.extend(product['health_assessment']['recommendations'])
        
        if all_recommendations:
            rec_counts = Counter(all_recommendations)
            
            st.write("**Top Recommendations:**")
            for rec, count in rec_counts.most_common(5):
                st.write(f"• {rec} (mentioned {count} times)")
        else:
            st.info("Add more products to get personalized recommendations!")
        
        # Category health analysis
        st.subheader("📊 Health by Category")
        
        category_health = {}
        for product in all_health_products:
            category = product['category']
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
                title="Average Health Score by Category (All Products)",
                labels={'x': 'Category', 'y': 'Average Health Score'},
                color=avg_scores,
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_category, use_container_width=True)
        
        # Product source breakdown
        st.subheader("📱 Product Source Analysis")
        
        source_counts = Counter([p['source'] for p in all_health_products])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Source distribution pie chart
            fig_source_pie = px.pie(
                values=list(source_counts.values()),
                names=list(source_counts.keys()),
                title="Products by Source",
                color_discrete_sequence=['#FF6B6B', '#4ECDC4']
            )
            st.plotly_chart(fig_source_pie, use_container_width=True)
        
        with col2:
            # Average health score by source
            source_health = {}
            for source in source_counts.keys():
                source_health[source] = sum([p['health_score'] for p in all_health_products if p['source'] == source]) / source_counts[source]
            
            fig_source_health = px.bar(
                x=list(source_health.keys()),
                y=list(source_health.values()),
                title="Average Health Score by Source",
                labels={'x': 'Product Source', 'y': 'Average Health Score'},
                color=list(source_health.values()),
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_source_health, use_container_width=True)
    
    else:
        st.info("No products to analyze yet! Add products to your cart or scan barcodes to see health insights.")
        
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
    st.subheader("🧪 Notify yourself")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(" 📩 Mail yourself"):
            if st.session_state.customer_profile['email']:
                test_message = f"""
Hello {st.session_state.customer_profile['name']},

This is a notification from your Smart Grocery Assistant 🛒.

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
                    "Notification - Smart Grocery Assistant 🛒",
                    test_message
                ):
                    st.success("Email sent successfully!")
                    add_notification("Email sent successfully", "success")
                else:
                    st.error("Failed to send  email.")
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
    <p>🛒 Smart Grocery Assistant</p>
    <p>Made by Harshada and Madhura</p>
    <p>Easy-Safe-Fast</p>
</div>
""", unsafe_allow_html=True)