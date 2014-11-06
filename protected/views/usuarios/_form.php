<style type="text/css">



    .secuencias{

        width: 200px;
        
        
    }



</style>

<p class="note">Campos con<span class="required">*</span> son necesarios.</p>

<?php echo $form->errorSummary($model); ?>

<div styler="width:470px !important;margin-left:0px;margin-bottom:5px !important;border-color:#961C1F;padding-top:10px;padding-right:5px;padding-left:14px; ">

<div>
    <div>
        <div>
            <div class="row">
		<?php echo $form->labelEx($model,'Username'); ?>
<?php echo $form->textField($model,'Username',array('size'=>60,'maxlength'=>128,'style' => 'width:200px')); ?>
<?php echo $form->error($model,'Username'); ?>
	</div>
            </div>
             <div>
                 <div class="row">
		<?php echo $form->labelEx($model,'Password'); ?>
<?php echo $form->passwordField($model,'Password',array('size'=>60,'maxlength'=>128,'style' => 'width:200px')); ?>
<?php echo $form->error($model,'Password'); ?>
	</div>
            </div>
        </div>
        <div>
        <div>
            <div class="row">
		<?php echo $form->labelEx($model,'Analista'); ?>
<?php echo $form->textField($model,'Analista',array('size'=>60,'maxlength'=>128,'style' => 'width:200px')); ?>
<?php echo $form->error($model,'Analista'); ?>
	</div>
            </div>
             <div>
            <div class="row">
		<?php echo $form->labelEx($model,'Proceso'); ?>
<?php echo $form->textField($model,'Proceso',array('size'=>60,'maxlength'=>128,'style' => 'width:200px')); ?>
<?php echo $form->error($model,'Proceso'); ?>
	</div>

             </div>
            </div>
            <div>
                
            <div>
                <div class="row">
			</div>
	<div class="row">
		<?php echo $form->labelEx($model,'Es_analista'); ?>
<?php echo $form->checkBox($model,'Es_analista'); ?>
<?php echo $form->error($model,'Es_analista'); ?>
	</div>
	<div class="row">
		<?php echo $form->labelEx($model,'Es_administrador'); ?>
<?php echo $form->checkBox($model,'Es_administrador'); ?>
<?php echo $form->error($model,'Es_administrador'); ?>
	</div>
                
                </div>
                <div>
                </div>
        </div>
    
    </div>
	
</div>
	

	

	

	



			