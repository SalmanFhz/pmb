import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import Counter

# Set page config
st.set_page_config(page_title="Student Registration Analysis", layout="wide")

@st.cache_data
def load_data(file):
    """Load and preprocess the data"""
    try:
        df = pd.read_csv(file, delimiter=';', quotechar='"', on_bad_lines='skip', encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

def clean_data(df):
    """Clean and preprocess the data"""
    # Clean salary columns by removing '\N' and converting to appropriate categories
    salary_columns = ['ayah_penghasilan', 'ibu_penghasilan']
    for col in salary_columns:
        df[col] = df[col].replace('\\N', '0 - 2.500.000')
    
    # Fill missing values
    df = df.fillna('Tidak Diketahui')
    
    return df

def create_salary_order():
    """Create ordered salary categories"""
    return ['0 - 2.500.000', '2.500.001 - 5.000.000', '5.000.001 - 7.500.000', 
            '7.500.001 - 10.000.000', '10.000.001 - 20.000.000', '20.000.001 - 50.000.000', 
            '50.000.001 - 100.000.000']

def demographic_analysis(df):
    """Perform demographic analysis"""
    st.header("📊 Analisis Demografis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution by domicile
        st.subheader("Distribusi Berdasarkan Domisili")
        domicile_counts = df['domisili'].value_counts()
        fig = px.pie(values=domicile_counts.values, names=domicile_counts.index,
                    title="Sebaran Domisili Calon Murid")
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.info(f"**Insight:** {domicile_counts.index[0]} mendominasi dengan {domicile_counts.iloc[0]} siswa ({domicile_counts.iloc[0]/len(df)*100:.1f}%)")
    
    with col2:
        # Distribution by category
        st.subheader("Distribusi Berdasarkan Kategori")
        category_counts = df['kategori'].value_counts()
        fig = px.bar(x=category_counts.index, y=category_counts.values,
                    title="Jumlah Calon Murid per Kategori")
        fig.update_layout(xaxis_title="Kategori", yaxis_title="Jumlah")
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"**Insight:** Mayoritas pendaftar adalah kategori {category_counts.index[0]} dengan {category_counts.iloc[0]} siswa")

def geographical_analysis(df):
    """Analyze geographical distribution"""
    st.header("🗺️ Analisis Geografis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Province distribution
        st.subheader("Sebaran Berdasarkan Provinsi")
        province_counts = df['alamat_propinsi'].value_counts()
        fig = px.bar(x=province_counts.values, y=province_counts.index,
                    orientation='h', title="Distribusi Provinsi Asal")
        fig.update_layout(yaxis_title="Provinsi", xaxis_title="Jumlah Siswa")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Regency distribution (top 10)
        st.subheader("Top 10 Kabupaten/Kota")
        regency_counts = df['alamat_kabupaten'].value_counts().head(10)
        fig = px.bar(x=regency_counts.values, y=regency_counts.index,
                    orientation='h', title="Top 10 Kabupaten/Kota Asal")
        fig.update_layout(yaxis_title="Kabupaten/Kota", xaxis_title="Jumlah Siswa")
        st.plotly_chart(fig, use_container_width=True)
    
    # Geographic insights
    st.info(f"""
    **Insights Geografis:**
    - Provinsi dominan: {province_counts.index[0]} ({province_counts.iloc[0]} siswa)
    - Kabupaten/Kota terbanyak: {regency_counts.index[0]} ({regency_counts.iloc[0]} siswa)
    - Total provinsi yang terwakili: {len(province_counts)} provinsi
    """)

def school_preference_analysis(df):
    """Analyze school preferences"""
    st.header("🎯 Analisis Preferensi Sekolah")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # First choice preferences
        st.subheader("Pilihan Pertama (Tujuan 1)")
        tujuan1_counts = df['tujuan1'].value_counts()
        fig = px.pie(values=tujuan1_counts.values, names=tujuan1_counts.index,
                    title="Distribusi Pilihan Pertama")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Campus preferences
        st.subheader("Preferensi Kampus")
        campus_cols = ['kampus1', 'kampus2', 'kampus3']
        all_campus = []
        for col in campus_cols:
            all_campus.extend(df[col].dropna().tolist())
        
        campus_counts = pd.Series(all_campus).value_counts()
        fig = px.bar(x=campus_counts.index, y=campus_counts.values,
                    title="Popularitas Kampus")
        fig.update_layout(xaxis_title="Kampus", yaxis_title="Jumlah Pilihan")
        st.plotly_chart(fig, use_container_width=True)
    
    st.success(f"""
    **Insights Preferensi:**
    - Pilihan pertama terpopuler: {tujuan1_counts.index[0]} ({tujuan1_counts.iloc[0]} siswa)
    - Kampus terfavorit: {campus_counts.index[0]} ({campus_counts.iloc[0]} pilihan)
    """)

def parent_education_analysis(df):
    """Analyze parent education levels"""
    st.header("🎓 Analisis Pendidikan Orang Tua")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Father's education
        st.subheader("Tingkat Pendidikan Ayah")
        ayah_edu = df['ayah_pendidikan'].value_counts()
        fig = px.pie(values=ayah_edu.values, names=ayah_edu.index,
                    title="Distribusi Pendidikan Ayah")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Mother's education
        st.subheader("Tingkat Pendidikan Ibu")
        ibu_edu = df['ibu_pendidikan'].value_counts()
        fig = px.pie(values=ibu_edu.values, names=ibu_edu.index,
                    title="Distribusi Pendidikan Ibu")
        st.plotly_chart(fig, use_container_width=True)
    
    # Education comparison
    st.subheader("Perbandingan Tingkat Pendidikan Orang Tua")
    education_comparison = pd.DataFrame({
        'Ayah': ayah_edu,
        'Ibu': ibu_edu
    }).fillna(0)
    
    fig = px.bar(education_comparison, 
                title="Perbandingan Tingkat Pendidikan Ayah vs Ibu")
    fig.update_layout(xaxis_title="Tingkat Pendidikan", yaxis_title="Jumlah")
    st.plotly_chart(fig, use_container_width=True)

def parent_occupation_analysis(df):
    """Analyze parent occupations"""
    st.header("💼 Analisis Pekerjaan Orang Tua")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Father's occupation
        st.subheader("Pekerjaan Ayah")
        ayah_job = df['ayah_pekerjaan'].value_counts().head(8)
        fig = px.bar(x=ayah_job.values, y=ayah_job.index,
                    orientation='h', title="Top 8 Pekerjaan Ayah")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Mother's occupation
        st.subheader("Pekerjaan Ibu")
        ibu_job = df['ibu_pekerjaan'].value_counts().head(8)
        fig = px.bar(x=ibu_job.values, y=ibu_job.index,
                    orientation='h', title="Top 8 Pekerjaan Ibu")
        st.plotly_chart(fig, use_container_width=True)
    
    # Occupation insights
    st.info(f"""
    **Insights Pekerjaan:**
    - Pekerjaan ayah terbanyak: {ayah_job.index[0]} ({ayah_job.iloc[0]} orang)
    - Pekerjaan ibu terbanyak: {ibu_job.index[0]} ({ibu_job.iloc[0]} orang)
    """)

def income_analysis(df):
    """Analyze parent income levels"""
    st.header("💰 Analisis Penghasilan Orang Tua")
    
    salary_order = create_salary_order()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Father's income
        st.subheader("Penghasilan Ayah")
        ayah_income = df['ayah_penghasilan'].value_counts()
        # Reorder based on salary order
        ayah_income_ordered = ayah_income.reindex([x for x in salary_order if x in ayah_income.index])
        
        fig = px.bar(x=ayah_income_ordered.index, y=ayah_income_ordered.values,
                    title="Distribusi Penghasilan Ayah")
        fig.update_layout(xaxis_title="Range Penghasilan", yaxis_title="Jumlah",
                         xaxis={'tickangle': 45})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Mother's income
        st.subheader("Penghasilan Ibu")
        ibu_income = df['ibu_penghasilan'].value_counts()
        ibu_income_ordered = ibu_income.reindex([x for x in salary_order if x in ibu_income.index])
        
        fig = px.bar(x=ibu_income_ordered.index, y=ibu_income_ordered.values,
                    title="Distribusi Penghasilan Ibu")
        fig.update_layout(xaxis_title="Range Penghasilan", yaxis_title="Jumlah",
                         xaxis={'tickangle': 45})
        st.plotly_chart(fig, use_container_width=True)
    
    # Combined family income analysis
    st.subheader("Analisis Gabungan Penghasilan Keluarga")
    
    # Calculate combined income categories
    def get_income_category(father_income, mother_income):
        father_max = get_max_income(father_income)
        mother_max = get_max_income(mother_income)
        total = father_max + mother_max
        
        if total <= 5000000:
            return "≤ 5 Juta"
        elif total <= 10000000:
            return "5-10 Juta"
        elif total <= 20000000:
            return "10-20 Juta"
        elif total <= 50000000:
            return "20-50 Juta"
        else:
            return "> 50 Juta"
    
    def get_max_income(income_range):
        if income_range == '0 - 2.500.000':
            return 2500000
        elif income_range == '2.500.001 - 5.000.000':
            return 5000000
        elif income_range == '5.000.001 - 7.500.000':
            return 7500000
        elif income_range == '7.500.001 - 10.000.000':
            return 10000000
        elif income_range == '10.000.001 - 20.000.000':
            return 20000000
        elif income_range == '20.000.001 - 50.000.000':
            return 50000000
        elif income_range == '50.000.001 - 100.000.000':
            return 100000000
        else:
            return 0
    
    df['combined_income_category'] = df.apply(
        lambda row: get_income_category(row['ayah_penghasilan'], row['ibu_penghasilan']), 
        axis=1
    )
    
    combined_income = df['combined_income_category'].value_counts()
    fig = px.pie(values=combined_income.values, names=combined_income.index,
                title="Estimasi Penghasilan Keluarga Gabungan")
    st.plotly_chart(fig, use_container_width=True)

def school_origin_analysis(df):
    """Analyze school origins"""
    st.header("🏫 Analisis Asal Sekolah")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # School origin by province
        st.subheader("Provinsi Asal Sekolah")
        school_province = df['propinsi_asal_sekolah'].value_counts()
        fig = px.pie(values=school_province.values, names=school_province.index,
                    title="Distribusi Provinsi Asal Sekolah")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top schools
        st.subheader("Top 10 Asal Sekolah")
        top_schools = df['asal_sekolah'].value_counts().head(10)
        fig = px.bar(x=top_schools.values, y=top_schools.index,
                    orientation='h', title="10 Sekolah Asal Terbanyak")
        st.plotly_chart(fig, use_container_width=True)

def summary_statistics(df):
    """Display summary statistics"""
    st.header("📈 Statistik Ringkasan")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Calon Murid", len(df))
    
    with col2:
        st.metric("Jumlah Provinsi", df['alamat_propinsi'].nunique())
    
    with col3:
        st.metric("Jumlah Sekolah Asal", df['asal_sekolah'].nunique())
    
    with col4:
        jawa_barat_count = len(df[df['domisili'] == 'JAWA BARAT'])
        st.metric("Siswa dari Jawa Barat", jawa_barat_count)
    
    # Additional insights
    st.subheader("Insights Utama")
    st.write("""
    **Key Findings dari Analisis Data:**
    
    1. **Dominasi Geografis**: Mayoritas calon murid berasal dari Jawa Barat, menunjukkan daya tarik sekolah di wilayah sekitar.
    
    2. **Preferensi Pendidikan**: Sebagian besar orang tua memiliki latar belakang pendidikan tinggi (S1/S2), menunjukkan kesadaran tinggi akan pentingnya pendidikan.
    
    3. **Diversitas Pekerjaan**: Orang tua memiliki beragam profesi dari PNS, swasta, hingga wirausaha, menunjukkan keberagaman latar belakang ekonomi.
    
    4. **Tingkat Ekonomi**: Distribusi penghasilan keluarga cukup beragam, dengan mayoritas berada di kelas menengah.
    
    5. **Pilihan Sekolah**: Preferensi terhadap jenjang SMP menunjukkan fokus pada pendidikan menengah pertama.
    """)

def main():
    st.title('📊 Dashboard Analisis Data Pendaftaran Siswa')
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header('⚙️ Pengaturan Analisis')
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Pilih file CSV data pendaftaran", 
        type=['csv'],
        help="Upload file CSV dengan format yang sesuai"
    )
    
    if uploaded_file is not None:
        try:
            # Load and clean data
            df = load_data(uploaded_file)
            if df is None:
                return
            df = clean_data(df)
            
            st.sidebar.success(f'✅ Data berhasil dimuat! ({len(df)} records)')
            
            # Sidebar options
            st.sidebar.subheader('Pilih Analisis')
            show_summary = st.sidebar.checkbox('Statistik Ringkasan', True)
            show_demographic = st.sidebar.checkbox('Analisis Demografis', True)
            show_geographic = st.sidebar.checkbox('Analisis Geografis', True)
            show_preferences = st.sidebar.checkbox('Analisis Preferensi Sekolah', True)
            show_parent_education = st.sidebar.checkbox('Pendidikan Orang Tua', True)
            show_parent_occupation = st.sidebar.checkbox('Pekerjaan Orang Tua', True)
            show_income = st.sidebar.checkbox('Analisis Penghasilan', True)
            show_school_origin = st.sidebar.checkbox('Asal Sekolah', True)
            
            # Main content
            if show_summary:
                summary_statistics(df)
                st.markdown("---")
            
            if show_demographic:
                demographic_analysis(df)
                st.markdown("---")
            
            if show_geographic:
                geographical_analysis(df)
                st.markdown("---")
            
            if show_preferences:
                school_preference_analysis(df)
                st.markdown("---")
            
            if show_parent_education:
                parent_education_analysis(df)
                st.markdown("---")
            
            if show_parent_occupation:
                parent_occupation_analysis(df)
                st.markdown("---")
            
            if show_income:
                income_analysis(df)
                st.markdown("---")
            
            if show_school_origin:
                school_origin_analysis(df)
                st.markdown("---")
            
            # Data preview
            with st.expander("👁️ Lihat Data Mentah"):
                st.dataframe(df)
                
                # Download processed data
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Data yang Telah Diproses",
                    data=csv,
                    file_name='processed_student_data.csv',
                    mime='text/csv'
                )
            
        except Exception as e:
            st.error(f'❌ Error memproses data: {str(e)}')
            st.info('Pastikan format CSV sesuai dengan template yang diharapkan')
    
    else:
        st.info('📁 Silakan upload file CSV data pendaftaran siswa untuk memulai analisis')
        
        # Show expected format
        with st.expander("📋 Format Data yang Diharapkan"):
            expected_columns = [
                'nama_calon_murid', 'kategori', 'jalur', 'tujuan1', 'tujuan2', 'tujuan3',
                'kampus1', 'kampus2', 'kampus3', 'domisili', 'alamat_propinsi',
                'alamat_kabupaten', 'asal_sekolah', 'propinsi_asal_sekolah',
                'ayah_pendidikan', 'ayah_pekerjaan', 'ayah_penghasilan',
                'ibu_pendidikan', 'ibu_pekerjaan', 'ibu_penghasilan'
            ]
            st.write("Kolom yang diperlukan:", expected_columns)

if __name__ == '__main__':
    main()
