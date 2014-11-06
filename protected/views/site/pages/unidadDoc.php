<?php $this->pageTitle=Yii::app()->name; ?>


<?php
$this->pageTitle=Yii::app()->name . ' - Administrar';
$this->breadcrumbs=array(
	'Administrar',
); ?>
<h><b>Por favor seleccione uno de los siguientes items para realizar operaciones (crear, borrar, editar, buscar o listar):</b></h>
        
	<br></br><a href="/index.php/metadocs/admin">documentos</a>
        <br></br><a href="/index.php/documentos/admin" >Documentos</a>
	<br></br><a href="/index.php/autorizaciones/admin">Autorizaciones</a>    
	<br></br><a href="/index.php/anotaciones/admin">Anotaciones</a>
	
