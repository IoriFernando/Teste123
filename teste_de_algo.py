"""
Navier-Stokes 2D incompressível - Método de projeção fracional
Baseado em Danaila et al. (2006) Capítulo 12

Implementações:
- Modo 1: Salva frames como PNG (para pós-processamento)
- Modo 2: Animação interativa com FuncAnimation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
from datetime import datetime

# ============================================================================
# ÍNDICES PERIÓDICOS (eq. 12.28-12.29)
# ============================================================================
def periodic_indices(n):
    """Retorna índices para condições de contorno periódicas"""
    ip = np.roll(np.arange(n), -1)  # i+1
    im = np.roll(np.arange(n),  1)  # i-1
    return ip, im


# ============================================================================
# THOMAS PARA SISTEMAS TRIDIAGONAIS (Algoritmo 12.5)
# ============================================================================
def thomas(aa, ab, ac, fi):
    """
    Resolve m sistemas tridiagonais em paralelo.
    
    Parâmetros:
    -----------
    aa, ab, ac, fi : arrays (m, n)
        aa: subdiagonal, ab: diagonal, ac: superdiagonal, fi: lado direito
    
    Retorna:
    --------
    x : array (m, n) solução
    """
    m, n = fi.shape
    b = ab.copy().astype(np.complex128)
    g = fi.copy().astype(np.complex128)
    
    # Forward sweep
    for k in range(1, n):
        w = aa[:, k] / b[:, k-1]
        b[:, k] -= w * ac[:, k-1]
        g[:, k] -= w * g[:, k-1]
    
    # Backward substitution
    x = np.zeros_like(g)
    x[:, -1] = g[:, -1] / b[:, -1]
    for k in range(n-2, -1, -1):
        x[:, k] = (g[:, k] - ac[:, k] * x[:, k+1]) / b[:, k]
    
    return x


def thomas_periodic(aa, ab, ac, fi, preserve_complex=True):
    """
    Resolve m sistemas tridiagonais PERIÓDICOS (Algoritmo 12.6)
    
    A matriz tem a forma:
    [b1 c1 0 ... 0 a1]
    [a2 b2 c2 ... 0 0]
    [0 a3 b3 ... 0 0]
    [...]
    [cn 0 0 ... an bn]
    
    Parâmetros:
    -----------
    aa, ab, ac, fi : arrays (m, n)
    preserve_complex : bool, se False retorna parte real
    
    Retorna:
    --------
    x : array (m, n) solução
    """
    m, n = fi.shape
    aa = aa.astype(np.complex128)
    ab = ab.astype(np.complex128)
    ac = ac.astype(np.complex128)
    fi = fi.astype(np.complex128)
    
    # Modificar diagonal para eliminar termos periódicos
    ab_s = ab.copy()
    ab_s[:, 0]  -= aa[:, 0]
    ab_s[:, -1] -= ac[:, -1]
    
    # Vetor para os termos periódicos
    v = np.zeros((m, n), dtype=np.complex128)
    v[:, 0]  = aa[:, 0]
    v[:, -1] = ac[:, -1]
    
    # Resolver dois sistemas auxiliares
    X1 = thomas(aa, ab_s, ac, fi)
    X2 = thomas(aa, ab_s, ac, v)
    
    # Calcular correção
    Xstar = (X1[:, 0] + X1[:, -1]) / (1.0 + X2[:, 0] + X2[:, -1])
    result = X1 - Xstar[:, np.newaxis] * X2
    
    return result if preserve_complex else result.real


# ============================================================================
# OPERADORES DIFERENCIAIS
# ============================================================================
def laplacian(f, ip, im, jp, jm, dx, dy):
    """
    Laplaciano discreto 2D com diferenças centradas (eq. 12.37)
    
    Δf = ∂²f/∂x² + ∂²f/∂y²
    """
    return ((f[ip, :] - 2*f + f[im, :]) / dx**2 + 
            (f[:, jp] - 2*f + f[:, jm]) / dy**2)


def gradient_x(f, im, dx):
    """Gradiente em x (diferença centrada)"""
    return (f - f[im, :]) / dx


def gradient_y(f, jm, dy):
    """Gradiente em y (diferença centrada)"""
    return (f - f[:, jm]) / dy


# ============================================================================
# (A) TERMOS EXPLÍCITOS DE CONVECÇÃO (eq. 12.31-12.32)
# ============================================================================
def explicit_terms(u, v, ip, im, jp, jm, dx, dy):
    """
    Calcula os termos convectivos H_u e H_v
    
    H_u = -(∂u²/∂x + ∂uv/∂y)  na posição u (xc_i, ym_j)
    H_v = -(∂uv/∂x + ∂v²/∂y)  na posição v (xm_i, yc_j)
    """
    nx = u.shape[0]
    ic = np.arange(nx)
    
    # ========== Hu: posição (xc_i, ym_j) ==========
    # ∂u²/∂x (usando interpolação para faces da célula)
    du2dx = (((u + u[ip, :]) / 2)**2 - ((u + u[im, :]) / 2)**2) / dx
    
    # ∂uv/∂y
    uv_north = (u + u[:, jp]) / 2 * (v + v[np.ix_(im, jp)]) / 2
    uv_south = (u + u[:, jm]) / 2 * (v + v[np.ix_(im, ic)]) / 2
    duv_dy = (uv_north - uv_south) / dy
    
    Hu = -(du2dx + duv_dy)
    
    # ========== Hv: posição (xm_i, yc_j) ==========
    # ∂v²/∂y
    dv2dy = (((v + v[:, jp]) / 2)**2 - ((v + v[:, jm]) / 2)**2) / dy
    
    # ∂uv/∂x
    uv_east = (u[np.ix_(ip, ic)] + u[np.ix_(ip, jm)]) / 2 * (v + v[ip, :]) / 2
    uv_west = (u + u[:, jm]) / 2 * (v + v[im, :]) / 2
    duv_dx = (uv_east - uv_west) / dx
    
    Hv = -(duv_dx + dv2dy)
    
    return Hu, Hv


# ============================================================================
# (B) ADI PARA EQUAÇÃO DE HELMHOLTZ (Algoritmo 12.2-12.3)
# ============================================================================
def adi_solve(rhs, beta_x, beta_y, nx, ny, dx, dy):
    """
    Resolve (I - βΔ)δq = rhs usando ADI com Thomas periódico
    
    Passos:
    1. (I - β_x ∂²/∂x²) δq̄ = rhs  (sweep em x)
    2. (I - β_y ∂²/∂y²) δq  = δq̄  (sweep em y)
    """
    # ========== Primeiro sweep: resolver em x para cada y ==========
    # Coeficientes para (I - β_x d²/dx²)
    ax = -beta_x * np.ones((ny, nx))  # subdiagonal
    bx = (1 + 2*beta_x) * np.ones((ny, nx))  # diagonal
    cx = -beta_x * np.ones((ny, nx))  # superdiagonal
    
    # RHS transposto para iterar sobre linhas
    dq_bar = thomas_periodic(ax, bx, cx, rhs.T, preserve_complex=False).T
    
    # ========== Segundo sweep: resolver em y para cada x ==========
    # Coeficientes para (I - β_y d²/dy²)
    ay = -beta_y * np.ones((nx, ny))  # subdiagonal
    by = (1 + 2*beta_y) * np.ones((nx, ny))  # diagonal
    cy = -beta_y * np.ones((nx, ny))  # superdiagonal
    
    dq = thomas_periodic(ay, by, cy, dq_bar, preserve_complex=False)
    
    return dq


# ============================================================================
# (C) SOLVER DE POISSON VIA FFT (Algoritmo 12.4)
# ============================================================================
def poisson_solve(Q, dx, dy):
    """
    Resolve Δφ = Q com condições periódicas
    
    Método:
    1. FFT em x para diagonalizar o Laplaciano
    2. Para cada modo, resolve sistema tridiagonal periódico em y
    3. IFFT para retornar ao espaço físico
    """
    nx, ny = Q.shape
    
    # FFT em x
    Q_hat = np.fft.fft(Q, axis=0)
    phi_hat = np.zeros_like(Q_hat, dtype=np.complex128)
    
    # Autovalores do Laplaciano em x (discretização centrada)
    kx = np.arange(nx)
    lambda_x = (2.0 / dx**2) * (np.cos(2 * np.pi * kx / nx) - 1)
    
    # Modo zero: fixa constante arbitrária
    phi_hat[0, :] = 0.0
    
    # Coeficientes para o sistema em y
    ay = (1.0 / dy**2) * np.ones((nx, ny))
    cy = (1.0 / dy**2) * np.ones((nx, ny))
    
    # Para cada modo em x (l = 1, ..., nx-1)
    for l in range(1, nx):
        # Diagonal principal para este modo
        by = (-2.0/dy**2 + lambda_x[l]) * np.ones(ny)
        
        # Resolver sistema tridiagonal periódico em y
        aa = ay[l, :][np.newaxis, :]
        bb = by[np.newaxis, :]
        cc = cy[l, :][np.newaxis, :]
        rhs = Q_hat[l, :][np.newaxis, :]
        
        phi_hat[l, :] = thomas_periodic(aa, bb, cc, rhs, preserve_complex=True)[0, :]
    
    # IFFT de volta
    return np.fft.ifft(phi_hat, axis=0).real


# ============================================================================
# VORTICIDADE (eq. 12.59-12.60)
# ============================================================================
def vorticity(u, v, im, jm, dx, dy):
    """
    Calcula a vorticidade ω = ∂v/∂x - ∂u/∂y
    
    ω(i,j) está em (xc_i, yc_j) - centro da célula
    """
    return (v - v[im, :]) / dx - (u - u[:, jm]) / dy


# ============================================================================
# (OPCIONAL) ESCALAR PASSIVO (eq. 12.61)
# ============================================================================
def advect_scalar(chi, u, v, ip, im, jp, jm, dx, dy, dt, Pe, chi_old_conv):
    """
    Avança o escalar passivo usando esquema Adams-Bashforth/Crank-Nicolson
    
    ∂χ/∂t + ∂(χu)/∂x + ∂(χv)/∂y = (1/Pe)Δχ
    """
    # Termo convectivo (Adams-Bashforth)
    conv = -(((chi + chi[ip, :])/2 * u[ip, :] - 
              (chi + chi[im, :])/2 * u) / dx +
             ((chi + chi[:, jp])/2 * v[:, jp] - 
              (chi + chi[:, jm])/2 * v) / dy)
    
    # Termo difusivo (Crank-Nicolson)
    lap_chi = laplacian(chi, ip, im, jp, jm, dx, dy)
    
    # RHS para Helmholtz
    rhs = dt * (1.5*conv - 0.5*chi_old_conv + lap_chi/Pe)
    
    # Coeficientes ADI para o escalar
    beta_x = dt / (2 * Pe * dx**2)
    beta_y = dt / (2 * Pe * dy**2)
    
    chi_new = chi + adi_solve(rhs, beta_x, beta_y, chi.shape[0], chi.shape[1], dx, dy)
    
    return chi_new, conv


# ============================================================================
# CONDIÇÕES INICIAIS
# ============================================================================
def ic_kelvin_helmholtz(xm, ym, Lx, Ly, U0=1.0, Pj=20.0, Ax=0.5, lam_x=None):
    """
    Kelvin-Helmholtz: perfil de jato com perturbação senoidal
    
    u(x,y) = u_mean(y) * (1 + u_pert(x))
    v(x,y) = 0
    
    O escalar passivo segue o perfil de velocidade inicial
    """
    if lam_x is None:
        lam_x = 0.5 * Lx
    
    X, Y = np.meshgrid(xm, ym, indexing='ij')
    Rj = Ly / 4.0  # Raio do jato
    
    # Perfil médio de velocidade (tangente hiperbólica)
    u_mean = U0/2 * (1 + np.tanh(0.5 * Pj * (1 - np.abs(Ly/2 - Y) / Rj)))
    
    # Perturbação senoidal para desencadear a instabilidade
    u_pert = Ax * np.sin(2 * np.pi * X / lam_x)
    
    u = u_mean * (1 + u_pert)
    v = np.zeros_like(u)
    p = np.zeros_like(u)
    chi = u_mean.copy()  # Escalar passivo inicialmente no jato
    
    return u, v, p, chi


def vortex_individual(xm, ym, xc, yc, psi0, lv):
    """
    Cria um vórtice individual via função corrente gaussiana
    
    ψ = ψ0 * exp(-r²/lv²)
    u = ∂ψ/∂y, v = -∂ψ/∂x
    """
    X, Y = np.meshgrid(xm, ym, indexing='ij')
    r2 = (X - xc)**2 + (Y - yc)**2
    psi = psi0 * np.exp(-r2 / lv**2)
    
    u = -2 * (Y - yc) / lv**2 * psi
    v =  2 * (X - xc) / lv**2 * psi
    
    return u, v


def ic_vortex_dipole(xm, ym, Lx, Ly, psi0=0.01, a=0.05):
    """
    Dipolo de vórtices: dois vórtices de sinais opostos
    
    Vórtice 1: +ψ0 em (Lx/4, Ly/2 + a)
    Vórtice 2: -ψ0 em (Lx/4, Ly/2 - a)
    
    O dipolo se propaga para a direita por auto-indução
    """
    nx, ny = len(xm), len(ym)
    xv = Lx / 4.0
    
    # Tamanho do vórtice (garante que caiba no domínio)
    lv = 0.4 * np.sqrt(2) * min(xv, Ly/2 - a, Lx - xv, Ly/2 - a)
    
    # Vórtice positivo (acima)
    u1, v1 = vortex_individual(xm, ym, xv, Ly/2 + a, +psi0, lv)
    
    # Vórtice negativo (abaixo)
    u2, v2 = vortex_individual(xm, ym, xv, Ly/2 - a, -psi0, lv)
    
    u = u1 + u2
    v = v1 + v2
    p = np.zeros((nx, ny))
    
    # Escalar passivo: faixa horizontal no centro
    chi = np.zeros((nx, ny))
    i_center = nx // 2
    width = max(1, nx // 12)  # Largura da faixa
    chi[i_center-width:i_center+width, :] = 1.0
    
    return u, v, p, chi


def ic_vortex_pair_interacting(xm, ym, Lx, Ly):
    """
    Dois dipolos colidindo frontalmente
    
    Dipolo 1: centro em x = Lx/4, propagando para direita
    Dipolo 2: centro em x = 3Lx/4, propagando para esquerda
    """
    # Primeiro dipolo (direita)
    u1, v1, _, _ = ic_vortex_dipole(xm, ym, Lx, Ly, psi0=0.008, a=0.06)
    
    # Segundo dipolo (esquerda) - deslocado e com vorticidade invertida
    # Para criar um dipolo que se move para esquerda, invertemos os sinais
    xv2 = 3*Lx/4
    lv = 0.4 * np.sqrt(2) * min(Lx/4, Ly/2-0.06, Lx/4, Ly/2-0.06)
    
    u_left1, v_left1 = vortex_individual(xm, ym, xv2, Ly/2 + 0.06, -0.008, lv)
    u_left2, v_left2 = vortex_individual(xm, ym, xv2, Ly/2 - 0.06, +0.008, lv)
    
    u2 = u_left1 + u_left2
    v2 = v_left1 + v_left2
    
    u = u1 + u2
    v = v1 + v2
    p = np.zeros((len(xm), len(ym)))
    chi = np.zeros((len(xm), len(ym)))
    
    # Duas faixas de escalar passivo
    i1 = len(xm) // 4
    i2 = 3 * len(xm) // 4
    width = len(xm) // 16
    chi[i1-width:i1+width, :] = 0.7
    chi[i2-width:i2+width, :] = 0.3
    
    return u, v, p, chi


# ============================================================================
# SOLVER PRINCIPAL - VERSÃO 1: SALVA FRAMES COMO PNG
# ============================================================================
def navier_stokes_save_frames(case='kh', nx=64, ny=64, Lx=2.0, Ly=1.0,
                               Re=1000.0, Pe=1000.0, cfl=0.35, t_end=10.0,
                               plot_interval=20, output_dir='output_frames',
                               verbose=True):
    """
    Resolve Navier-Stokes e salva frames como imagens PNG
    
    Retorna:
    --------
    frames : lista de tuplas (fig, t) - figuras salvas
    history : dicionário com evolução temporal dos campos
    """
    
    # Criar diretório de saída
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dx = Lx / nx
    dy = Ly / ny
    
    # Malha staggered
    xm = (np.arange(nx) + 0.5) * dx
    ym = (np.arange(ny) + 0.5) * dy
    
    # Índices periódicos
    ip, im = periodic_indices(nx)
    jp, jm = periodic_indices(ny)
    
    # Condição inicial
    if case == 'kh':
        u, v, p, chi = ic_kelvin_helmholtz(xm, ym, Lx, Ly)
        title = "Kelvin-Helmholtz Instability"
    elif case == 'dipole':
        u, v, p, chi = ic_vortex_dipole(xm, ym, Lx, Ly)
        title = "Vortex Dipole"
    elif case == 'interaction':
        u, v, p, chi = ic_vortex_pair_interacting(xm, ym, Lx, Ly)
        title = "Vortex Dipole Interaction"
    else:
        raise ValueError("case deve ser 'kh', 'dipole' ou 'interaction'")
    
    # Time step baseado na CFL
    vel_max = np.max(np.abs(u)/dx + np.abs(v)/dy)
    dt = cfl / vel_max if vel_max > 0 else 0.001
    
    # Coeficientes ADI (constantes no tempo)
    beta_x_mom = dt / (2 * Re * dx**2)
    beta_y_mom = dt / (2 * Re * dy**2)
    
    # Gradiente de pressão inicial
    dpdx = gradient_x(p, im, dx)
    dpdy = gradient_y(p, jm, dy)
    
    # Termos convectivos do passo anterior (Adams-Bashforth)
    Hu_old = np.zeros((nx, ny))
    Hv_old = np.zeros((nx, ny))
    chi_conv_old = np.zeros((nx, ny))
    
    # Histórico para animação
    history = {
        't': [0.0],
        'omega': [vorticity(u, v, im, jm, dx, dy).copy()],
        'chi': [chi.copy()],
        'u': [u.copy()],
        'v': [v.copy()]
    }
    
    t = 0.0
    step = 0
    frames = []
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Navier-Stokes 2D - {title}")
        print(f"{'='*70}")
        print(f"Malha: {nx} x {ny} | Lx={Lx} Ly={Ly}")
        print(f"Re={Re} | Pe={Pe} | dt={dt:.5f} | cfl={cfl}")
        print(f"t_end={t_end} | ~{int(t_end/dt)} passos")
        print(f"Salvando frames em: {output_dir}")
        print(f"{'='*70}\n")
    
    while t < t_end - 1e-10:
        # ========== (A) Termos convectivos explícitos ==========
        Hu, Hv = explicit_terms(u, v, ip, im, jp, jm, dx, dy)
        
        # ========== (B) Helmholtz - passo preditor ==========
        lap_u = laplacian(u, ip, im, jp, jm, dx, dy)
        lap_v = laplacian(v, ip, im, jp, jm, dx, dy)
        
        rhs_u = dt * (-dpdx + 1.5*Hu - 0.5*Hu_old + lap_u/(2*Re))
        rhs_v = dt * (-dpdy + 1.5*Hv - 0.5*Hv_old + lap_v/(2*Re))
        
        u_star = u + adi_solve(rhs_u, beta_x_mom, beta_y_mom, nx, ny, dx, dy)
        v_star = v + adi_solve(rhs_v, beta_x_mom, beta_y_mom, nx, ny, dx, dy)
        
        # ========== (C) Poisson para correção de pressão ==========
        div_star = (u_star[ip, :] - u_star) / dx + (v_star[:, jp] - v_star) / dy
        Q = div_star / dt
        
        phi = poisson_solve(Q, dx, dy)
        
        # ========== (D) Correção da velocidade ==========
        u = u_star - dt * gradient_x(phi, im, dx)
        v = v_star - dt * gradient_y(phi, jm, dy)
        
        # ========== (E) Atualização da pressão ==========
        lap_phi = laplacian(phi, ip, im, jp, jm, dx, dy)
        p = p + phi - (dt / (2*Re)) * lap_phi
        
        # Atualiza gradiente de pressão
        dpdx = gradient_x(p, im, dx)
        dpdy = gradient_y(p, jm, dy)
        
        # ========== Escalar passivo ==========
        chi, chi_conv_old = advect_scalar(chi, u, v, ip, im, jp, jm,
                                           dx, dy, dt, Pe, chi_conv_old)
        
        # Atualiza termos do passo anterior
        Hu_old, Hv_old = Hu, Hv
        
        t += dt
        step += 1
        
        # ========== Diagnóstico ==========
        if step % 50 == 0 and verbose:
            div_max = np.max(np.abs((u[ip, :]-u)/dx + (v[:, jp]-v)/dy))
            omega_rms = np.sqrt(np.mean(vorticity(u, v, im, jm, dx, dy)**2))
            ke = 0.5 * np.mean(u**2 + v**2)  # Energia cinética
            print(f"  passo {step:6d} | t={t:6.3f} | div_max={div_max:.2e} | "
                  f"ω_rms={omega_rms:.4f} | KE={ke:.4f}")
        
        # ========== Salvar frame ==========
        if step % plot_interval == 0 or t >= t_end:
            omega = vorticity(u, v, im, jm, dx, dy)
            X, Y = np.meshgrid(xm, ym, indexing='ij')
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
            
            # Vorticidade
            cf1 = axes[0].contourf(X, Y, omega, levels=41, cmap='RdBu_r')
            plt.colorbar(cf1, ax=axes[0], label='Vorticidade ω')
            axes[0].set_title(f'Vorticidade - t = {t:.2f}', fontsize=12)
            axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
            axes[0].set_aspect('equal')
            
            # Escalar passivo
            cf2 = axes[1].contourf(X, Y, chi, levels=41, cmap='viridis')
            plt.colorbar(cf2, ax=axes[1], label='Escalar χ')
            axes[1].set_title(f'Escalar passivo - t = {t:.2f}', fontsize=12)
            axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
            axes[1].set_aspect('equal')
            
            plt.suptitle(f'{title} | Re={Re}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Salvar figura
            fname = os.path.join(output_dir, f'{case}_step{step:06d}.png')
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            frames.append(fname)
            
            # Armazenar histórico
            history['t'].append(t)
            history['omega'].append(omega.copy())
            history['chi'].append(chi.copy())
            history['u'].append(u.copy())
            history['v'].append(v.copy())
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Simulação concluída: {step} passos, t_final={t:.4f}")
        print(f"Frames salvos: {len(frames)}")
        print(f"{'='*70}\n")
    
    return frames, history


# ============================================================================
# SOLVER PRINCIPAL - VERSÃO 2: ANIMAÇÃO INTERATIVA (FuncAnimation)
# ============================================================================
def navier_stokes_animate(case='kh', nx=64, ny=64, Lx=2.0, Ly=1.0,
                           Re=1000.0, Pe=1000.0, cfl=0.35, t_end=10.0,
                           plot_interval=20, save_video=False, 
                           video_name='ns_animation.mp4', verbose=True):
    """
    Resolve Navier-Stokes e cria animação interativa com FuncAnimation
    
    Parâmetros:
    -----------
    save_video : bool, se True salva como arquivo de vídeo
    video_name : str, nome do arquivo de vídeo
    
    Retorna:
    --------
    anim : FuncAnimation object
    history : dicionário com evolução temporal
    """
    
    dx = Lx / nx
    dy = Ly / ny
    
    # Malha staggered
    xm = (np.arange(nx) + 0.5) * dx
    ym = (np.arange(ny) + 0.5) * dy
    
    # Índices periódicos
    ip, im = periodic_indices(nx)
    jp, jm = periodic_indices(ny)
    
    # Condição inicial
    if case == 'kh':
        u, v, p, chi = ic_kelvin_helmholtz(xm, ym, Lx, Ly)
        title = "Kelvin-Helmholtz Instability"
    elif case == 'dipole':
        u, v, p, chi = ic_vortex_dipole(xm, ym, Lx, Ly)
        title = "Vortex Dipole"
    elif case == 'interaction':
        u, v, p, chi = ic_vortex_pair_interacting(xm, ym, Lx, Ly)
        title = "Vortex Dipole Interaction"
    else:
        raise ValueError("case deve ser 'kh', 'dipole' ou 'interaction'")
    
    # Time step
    vel_max = np.max(np.abs(u)/dx + np.abs(v)/dy)
    dt = cfl / vel_max if vel_max > 0 else 0.001
    
    # Coeficientes ADI
    beta_x_mom = dt / (2 * Re * dx**2)
    beta_y_mom = dt / (2 * Re * dy**2)
    
    # Estado inicial
    dpdx = gradient_x(p, im, dx)
    dpdy = gradient_y(p, jm, dy)
    Hu_old = np.zeros((nx, ny))
    Hv_old = np.zeros((nx, ny))
    chi_conv_old = np.zeros((nx, ny))
    
    # Histórico para animação
    history = {
        't': [0.0],
        'omega': [vorticity(u, v, im, jm, dx, dy).copy()],
        'chi': [chi.copy()],
        'u': [u.copy()],
        'v': [v.copy()]
    }
    
    # Armazenar estado atual para uso na animação
    state = {
        'u': u, 'v': v, 'p': p, 'chi': chi,
        'dpdx': dpdx, 'dpdy': dpdy,
        'Hu_old': Hu_old, 'Hv_old': Hv_old,
        'chi_conv_old': chi_conv_old,
        't': 0.0, 'step': 0
    }
    
    def advance_one_step():
        """Avança um passo de tempo e retorna o novo estado"""
        u = state['u']; v = state['v']; p = state['p']; chi = state['chi']
        dpdx = state['dpdx']; dpdy = state['dpdy']
        Hu_old = state['Hu_old']; Hv_old = state['Hv_old']
        chi_conv_old = state['chi_conv_old']
        
        # (A) Convecção
        Hu, Hv = explicit_terms(u, v, ip, im, jp, jm, dx, dy)
        
        # (B) Helmholtz
        lap_u = laplacian(u, ip, im, jp, jm, dx, dy)
        lap_v = laplacian(v, ip, im, jp, jm, dx, dy)
        
        rhs_u = dt * (-dpdx + 1.5*Hu - 0.5*Hu_old + lap_u/(2*Re))
        rhs_v = dt * (-dpdy + 1.5*Hv - 0.5*Hv_old + lap_v/(2*Re))
        
        u_star = u + adi_solve(rhs_u, beta_x_mom, beta_y_mom, nx, ny, dx, dy)
        v_star = v + adi_solve(rhs_v, beta_x_mom, beta_y_mom, nx, ny, dx, dy)
        
        # (C) Poisson
        div_star = (u_star[ip, :] - u_star) / dx + (v_star[:, jp] - v_star) / dy
        phi = poisson_solve(div_star / dt, dx, dy)
        
        # (D) Correção
        u_new = u_star - dt * gradient_x(phi, im, dx)
        v_new = v_star - dt * gradient_y(phi, jm, dy)
        
        # (E) Pressão
        lap_phi = laplacian(phi, ip, im, jp, jm, dx, dy)
        p_new = p + phi - (dt / (2*Re)) * lap_phi
        dpdx_new = gradient_x(p_new, im, dx)
        dpdy_new = gradient_y(p_new, jm, dy)
        
        # Escalar passivo
        chi_new, chi_conv_new = advect_scalar(chi, u_new, v_new, ip, im, jp, jm,
                                               dx, dy, dt, Pe, chi_conv_old)
        
        # Atualizar estado
        state['u'] = u_new; state['v'] = v_new; state['p'] = p_new
        state['chi'] = chi_new
        state['dpdx'] = dpdx_new; state['dpdy'] = dpdy_new
        state['Hu_old'] = Hu; state['Hv_old'] = Hv
        state['chi_conv_old'] = chi_conv_new
        state['t'] += dt
        state['step'] += 1
        
        # Armazenar histórico (a cada plot_interval)
        if state['step'] % plot_interval == 0:
            omega = vorticity(u_new, v_new, im, jm, dx, dy)
            history['t'].append(state['t'])
            history['omega'].append(omega.copy())
            history['chi'].append(chi_new.copy())
            history['u'].append(u_new.copy())
            history['v'].append(v_new.copy())
        
        return state['t'], state['step']
    
    # Executar simulação e coletar histórico
    if verbose:
        print(f"\n{'='*70}")
        print(f"Navier-Stokes 2D (Animação) - {title}")
        print(f"{'='*70}")
        print(f"Malha: {nx} x {ny} | Re={Re} | dt={dt:.5f}")
        print(f"t_end={t_end} | Coletando frames a cada {plot_interval} passos")
        print(f"{'='*70}\n")
    
    while state['t'] < t_end - 1e-10:
        t_curr, step_curr = advance_one_step()
        if step_curr % 100 == 0 and verbose:
            print(f"  passo {step_curr:6d} | t={t_curr:6.3f}")
    
    if verbose:
        print(f"\nSimulação concluída: {state['step']} passos, t_final={state['t']:.4f}")
        print(f"Frames coletados: {len(history['t'])}\n")
    
    # ========== Criar animação ==========
    X, Y = np.meshgrid(xm, ym, indexing='ij')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f'{title} | Re={Re}', fontsize=14, fontweight='bold')
    
    # Configuração dos eixos
    for ax in axes:
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_aspect('equal')
    
    # Níveis de contorno
    omega_levels = np.linspace(-np.max(np.abs(history['omega'][0])), 
                                np.max(np.abs(history['omega'][0])), 31)
    chi_levels = 41
    
    # Primeiro frame
    cf1 = axes[0].contourf(X, Y, history['omega'][0], levels=omega_levels, cmap='RdBu_r')
    cf2 = axes[1].contourf(X, Y, history['chi'][0], levels=chi_levels, cmap='viridis')
    
    plt.colorbar(cf1, ax=axes[0], label='Vorticidade ω')
    plt.colorbar(cf2, ax=axes[1], label='Escalar χ')
    
    axes[0].set_title(f'Vorticidade - t = {history["t"][0]:.2f}')
    axes[1].set_title(f'Escalar passivo - t = {history["t"][0]:.2f}')
    
    def update_frame(frame):
        """Atualiza o frame da animação"""
        for ax in axes:
            for c in ax.collections:
                c.remove()
        
        omega = history['omega'][frame]
        chi = history['chi'][frame]
        
        # Atualizar níveis automaticamente para vorticidade
        omega_max = np.max(np.abs(omega))
        levels_omega = np.linspace(-omega_max, omega_max, 31) if omega_max > 0 else [-1, 1]
        
        cf1 = axes[0].contourf(X, Y, omega, levels=levels_omega, cmap='RdBu_r')
        cf2 = axes[1].contourf(X, Y, chi, levels=41, cmap='viridis')
        
        axes[0].set_title(f'Vorticidade - t = {history["t"][frame]:.2f}')
        axes[1].set_title(f'Escalar passivo - t = {history["t"][frame]:.2f}')
        
        return cf1, cf2
    
    anim = FuncAnimation(fig, update_frame, frames=len(history['t']), 
                         interval=50, blit=False, repeat=True)
    
    if save_video:
        print(f"Salvando vídeo: {video_name}")
        anim.save(video_name, writer=PillowWriter(fps=20))
        print(f"Vídeo salvo com sucesso!")
    
    plt.tight_layout()
    
    return anim, history


# ============================================================================
# FUNÇÕES DE VISUALIZAÇÃO E PÓS-PROCESSAMENTO
# ============================================================================
def create_mosaic_from_frames(output_dir, case_name, nrows=3, ncols=3):
    """
    Cria um mosaico de frames para visualização rápida
    
    Parâmetros:
    -----------
    output_dir : str, diretório com os frames PNG
    case_name : str, nome do caso
    nrows, ncols : int, dimensões do mosaico
    """
    import glob
    
    frames = sorted(glob.glob(os.path.join(output_dir, f'{case_name}_step*.png')))
    
    if len(frames) == 0:
        print(f"Nenhum frame encontrado em {output_dir}")
        return
    
    # Selecionar frames igualmente espaçados
    indices = np.linspace(0, len(frames)-1, nrows*ncols, dtype=int)
    selected_frames = [frames[i] for i in indices]
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = axes.flatten() if nrows*ncols > 1 else [axes]
    
    for idx, (ax, fpath) in enumerate(zip(axes, selected_frames)):
        img = plt.imread(fpath)
        ax.imshow(img)
        ax.axis('off')
        # Extrair tempo do nome do arquivo
        step = int(fpath.split('_step')[-1].split('.')[0])
        ax.set_title(f'step={step}', fontsize=10)
    
    plt.suptitle(f'{case_name.upper()} - Evolution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    mosaic_path = os.path.join(output_dir, f'{case_name}_mosaic.png')
    plt.savefig(mosaic_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Mosaico salvo: {mosaic_path}")


def plot_energy_history(history, output_dir='.'):
    """
    Plota evolução da energia cinética e enstrofia
    
    Energia cinética: E = ½∫(u²+v²) dΩ
    Enstrofia: Ω = ½∫ω² dΩ
    """
    t = np.array(history['t'])
    
    energy = []
    enstrophy = []
    
    for i in range(len(t)):
        u = history['u'][i]
        v = history['v'][i]
        omega = history['omega'][i]
        
        energy.append(0.5 * np.mean(u**2 + v**2))
        enstrophy.append(0.5 * np.mean(omega**2))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(t, energy, 'b-', linewidth=2)
    ax1.set_xlabel('Tempo t')
    ax1.set_ylabel('Energia cinética E')
    ax1.set_title('Evolução da Energia')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t, enstrophy, 'r-', linewidth=2)
    ax2.set_xlabel('Tempo t')
    ax2.set_ylabel('Enstrofia Ω')
    ax2.set_title('Evolução da Enstrofia')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_history.png'), dpi=150)
    plt.close()
    print("Gráfico de energia salvo: energy_history.png")


# ============================================================================
# MAIN - EXECUÇÃO
# ============================================================================
if __name__ == '__main__':
    
    print("="*70)
    print("NAVIER-STOKES 2D INCOMPRESSÍVEL")
    print("Baseado em Danaila et al. (2006) - Capítulo 12")
    print("="*70)
    
    # ========== EXEMPLO 1: Salvar frames como PNG ==========
    print("\n[1] Executando simulação com salvamento de frames...")
    
    frames_kh, history_kh = navier_stokes_save_frames(
        case='kh',           # Kelvin-Helmholtz
        nx=96, ny=96,        # Resolução
        Lx=2.0, Ly=1.0,      # Domínio
        Re=1000, Pe=1000,    # Números adimensionais
        cfl=0.35,            # Fator CFL (estabilidade)
        t_end=8.0,           # Tempo final
        plot_interval=25,    # Salvar a cada 25 passos
        output_dir='output_kh',
        verbose=True
    )
    
    # Criar mosaico dos frames
    create_mosaic_from_frames('output_kh', 'kh', nrows=2, ncols=4)
    
    # Plotar evolução da energia
    plot_energy_history(history_kh, 'output_kh')
    
    # ========== EXEMPLO 2: Dipolo de vórtice ==========
    print("\n[2] Executando simulação do dipolo de vórtice...")
    
    frames_dip, history_dip = navier_stokes_save_frames(
        case='dipole',
        nx=96, ny=96,
        Lx=1.0, Ly=1.0,
        Re=1000, Pe=1000,
        cfl=0.35,
        t_end=6.0,
        plot_interval=20,
        output_dir='output_dipole',
        verbose=True
    )
    
    create_mosaic_from_frames('output_dipole', 'dipole', nrows=2, ncols=4)
    plot_energy_history(history_dip, 'output_dipole')
    
    # ========== EXEMPLO 3: Animação interativa ==========
    print("\n[3] Criando animação interativa (FuncAnimation)...")
    
    anim, hist_interact = navier_stokes_animate(
        case='dipole',
        nx=64, ny=64,
        Lx=1.0, Ly=1.0,
        Re=1000, Pe=1000,
        cfl=0.35,
        t_end=5.0,
        plot_interval=15,
        save_video=True,
        video_name='vortex_dipole.gif',
        verbose=True
    )
    
    # Mostrar animação (se estiver em ambiente interativo)
    # plt.show()  # Descomente para exibir
    
    # ========== EXEMPLO 4: Interação entre dipolos ==========
    print("\n[4] Executando simulação de interação entre dipolos...")
    
    frames_int, history_int = navier_stokes_save_frames(
        case='interaction',
        nx=128, ny=128,
        Lx=2.0, Ly=1.0,
        Re=2000, Pe=2000,
        cfl=0.3,
        t_end=8.0,
        plot_interval=30,
        output_dir='output_interaction',
        verbose=True
    )
    
    create_mosaic_from_frames('output_interaction', 'interaction', nrows=2, ncols=4)
    
    print("\n" + "="*70)
    print("TODAS AS SIMULAÇÕES CONCLUÍDAS COM SUCESSO!")
    print("="*70)
    print("\nArquivos gerados:")
    print("  - output_kh/        : Frames e mosaico do caso Kelvin-Helmholtz")
    print("  - output_dipole/    : Frames e mosaico do dipolo de vórtice")
    print("  - output_interaction/: Frames do caso de interação")
    print("  - vortex_dipole.gif : Animação do dipolo")
    print("  - energy_history.png: Evolução da energia/enstrofia")
    print("="*70)