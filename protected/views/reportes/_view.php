<div class="view">

        <b><?php echo CHtml::encode($data->getAttributeLabel('Fecha')); ?>:</b>
	<?php echo CHtml::link(CHtml::encode($data->Fecha), array('view', 'id'=>$data->id)); ?>
	<br />	

	<b><?php echo CHtml::encode($data->getAttributeLabel('Presion')); ?>:</b>
	<?php echo CHtml::encode($data->Presion); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Decibeles')); ?>:</b>
	<?php echo CHtml::encode($data->Decibeles); ?>
	<br />	
        <b><?php 
    
        
        echo CHtml::encode($data->getAttributeLabel('Estado')); ?>:</b>
	<?php 
        $EstadoIn=$data->Estado;
        if ($EstadoIn==0) print('<img src="/images/verde.gif" height="15" width="15" /> 0 - Adecuado');
        if ($EstadoIn==1) print('<img src="/images/amarillo.gif" height="15" width="15" /> 1 - Posible deficiencia - Se requiere más información.');
        if ($EstadoIn==2) print('<img src="/images/amarillo.gif" height="15" width="15" /> 2 - Deficiencia - Reparar Inmediatamente');
        if ($EstadoIn==3) print('<img src="/images/rojo.gif" height="15" width="15" /> 3 - Deficiencia - Reparar Inmediatamente');
        if ($EstadoIn==4) print('<img src="/images/rojo.gif" height="15" width="15" /> 4 - Deficiencia Grave - Inmediatamente');
           
           ?>
	<br />


	<?php /*
	<b><?php echo CHtml::encode($data->getAttributeLabel('Proceso')); ?>:</b>
	<?php echo CHtml::encode($data->Proceso); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Area')); ?>:</b>
	<?php echo CHtml::encode($data->Area); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Equipo')); ?>:</b>
	<?php echo CHtml::encode($data->Equipo); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Analista')); ?>:</b>
	<?php echo CHtml::encode($data->Analista); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('OT')); ?>:</b>
	<?php echo CHtml::encode($data->OT); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Fecha')); ?>:</b>
	<?php echo CHtml::encode($data->Fecha); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Gas')); ?>:</b>
	<?php echo CHtml::encode($data->Gas); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Tamano')); ?>:</b>
	<?php echo CHtml::encode($data->Tamano); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('CFM')); ?>:</b>
	<?php echo CHtml::encode($data->CFM); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('COSTO')); ?>:</b>
	<?php echo CHtml::encode($data->COSTO); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Corregido')); ?>:</b>
	<?php echo CHtml::encode($data->Corregido); ?>
	<br />

	*/ ?>

</div>
