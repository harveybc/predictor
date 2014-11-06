<style type="text/css">



    .secuencias{

        width: 200px;
        
        
    }

</style>
<div styler="width:700px !important;height:300px;margin-left:0px;margin-bottom:5px !important;border-color:#961C1F;padding-top:10px;padding-right:5px;padding-left:14px; ">

<p class="note">Campos con<span class="required">*</span> son necesarios.</p>

<?php echo $form->errorSummary($model); ?>

	<div>
            <div>
                <div>
                    <div>
                    <?php echo $form->labelEx($model,'cedula'); ?>
<?php echo $form->textField($model,'cedula',array('size'=>32,'maxlength'=>32)); ?>
<?php echo $form->error($model,'cedula'); ?>
	</div>
                    </div>
                    <div>
                        <div class="row">
		<?php echo $form->labelEx($model,'fechaPrestamo'); ?>
<?php echo $form->textField($model,'fechaPrestamo'); ?>
<?php echo $form->error($model,'fechaPrestamo'); ?>
	</div>
                    </div>
                    <div>
                        <div class="row">
		<?php echo $form->labelEx($model,'fechaDevolucion'); ?>
<?php echo $form->textField($model,'fechaDevolucion'); ?>
<?php echo $form->error($model,'fechaDevolucion'); ?>
	</div>
                    </div>
                    </div>
                    <div>
                        <div>
                        <div>
                    <div>
                        <div class="row">
		<?php echo $form->labelEx($model,'observaciones'); ?>
<?php echo $form->textField($model,'observaciones',array('size'=>60,'maxlength'=>128)); ?>
<?php echo $form->error($model,'observaciones'); ?>
	</div>
                    </div>
                    
                </div>
            </div>
		

<div>
    <div>
        <div>
            <label for="MetaDocs">documentos prestados</label><?php 
					$this->widget('application.components.Relation', array(
							'model' => $model,
							'relation' => 'metaDoc0',
							'fields' => 'numPedido',
							'allowEmpty' => true,
							'style' => 'dropdownlist',
                                                        'htmlOptions' => array(
                                                                'class' => 'secuencias',)
							)
						); ?>
            </div>
            <div>
                <label for="Usuarios">Usuarios que realizaron el préstamo</label><?php 
					$this->widget('application.components.Relation', array(
							'model' => $model,
							'relation' => 'usuario0',
							'fields' => 'Username',
							'allowEmpty' => false,
							'style' => 'dropdownlist',
                                                        'htmlOptions' => array(
                                                                'class' => 'secuencias',)
							)
						); ?>
            </div>
            <div>
                <label for="Usuarios">Usuarios que solicitaron el préstamo</label><?php 
					$this->widget('application.components.Relation', array(
							'model' => $model,
							'relation' => 'usuarioRcv0',
							'fields' => 'Username',
							'allowEmpty' => false,
							'style' => 'dropdownlist',
                                                        'htmlOptions' => array(
                                                                'class' => 'secuencias',)
							)
						); ?>
            </div>
        </div>
    </div>
</div>

			
			
			