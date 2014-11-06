<?php $this->pageTitle=Yii::app()->name; ?>


<?php
$this->pageTitle=Yii::app()->name . ' - Administrar';
$this->breadcrumbs=array(
	'Administrar',
); ?>
<h><b>Por favor seleccione uno de los siguientes items para realizar operaciones (crear, borrar, editar, buscar o listar):</b></h>
        
        <br></br><a href="/index.php/motores/admin" >Base de Datos Motores</a>
	<br></br><a href="/index.php/tableros/admin">Base de Datos Tableros</a>
        <br></br><a href="/index.php/usuarios/admin" >Analistas</a>
        <br></br><a href="/index.php/site/page?view=tableroVirtualCbm" >Tablero Virtual CBM</a>
        <br></br><a href="/index.php/site/page?view=backup" >Backup de BD</a>
	<br></br><a href="/index.php/estructura/admin">Equipos</a>
        <br></br><a href="/index.php/site/page?view=buscarOT" >Buscar por Orden de Trabajo</a>
        
	
	
